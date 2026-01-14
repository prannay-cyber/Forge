use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::{Output, tools};
use std::pin::Pin;
use std::future::Future;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: Vec<String>,
    pub description: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<Message>,
}

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<Content>,
}

#[derive(Deserialize)]
struct Content {
    text: String,
}

pub struct Agent {
    client: Client,
    api_key: String,
    output: Output,
    system_prompt: String,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(api_key: String) -> Self {
        let system_prompt = "\
You are Forge, an advanced AI coding agent with access to file system tools.

Your capabilities:
- read <path>: Read file contents
- write <path> <content>: Write/create files
- edit <path> <search> <replace>: Edit files
- bash <command>: Execute shell commands
- glob <pattern>: Find files by pattern
- grep <pattern> <path>: Search file contents

Your behavior:
- You maintain full conversation context across all interactions
- When asked for summaries, you reference previous actions and results
- You think step-by-step about problems before acting
- When you encounter errors, you analyze what went wrong and try different approaches
- You are concise but thorough
- You execute tasks autonomously without asking for permission unless truly ambiguous

Current working directory is preserved across commands.".to_string();

        Self {
            client: Client::new(),
            api_key,
            output: Output::new(),
            system_prompt,
            messages: Vec::new(),
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: content.to_string(),
        });
    }

    pub async fn process(&mut self) -> Result<()> {
        const MAX_TURNS: u32 = 5;

        for turn in 1..=MAX_TURNS {
            if turn > 1 {
                self.output.tool_header(&format!("Turn {}", turn));
                println!();
            }

            // Step 1: Think and plan
            self.output.tool_header("Thinking");

            let plan_prompt = if turn == 1 {
                "\
Based on the conversation above, think step-by-step:
1. What is the user asking for?
2. What tools do I need to use?
3. What's my approach?

Then output tool calls as a JSON array at the end.

Available tools:
- read <path>: Read a file
- write <path> <content>: Write a file
- edit <path> <search> <replace>: Edit a file
- bash <command>: Run shell command
- glob <pattern>: Find files
- grep <pattern> <path>: Search contents

Format your response as:
[Your reasoning here]

TOOL_CALLS:
[{\"tool\": \"read\", \"args\": [\"path\"], \"description\": \"what this does\"}]

If no tools are needed (e.g., just answering a question), respond with your answer and end with:
TOOL_CALLS:
[]"
            } else {
                "\
Based on the tool results above, think step-by-step:
1. Did the previous tools give me what I need?
2. Is the task complete, or do I need to do more?
3. If more work is needed, what tools should I use next?

Available tools:
- read <path>: Read a file
- write <path> <content>: Write a file
- edit <path> <search> <replace>: Edit a file
- bash <command>: Run shell command
- glob <pattern>: Find files
- grep <pattern> <path>: Search contents

Format:
[Your reasoning - explain if task is complete or what to do next]

TOOL_CALLS:
[{\"tool\": \"read\", \"args\": [\"path\"], \"description\": \"what this does\"}]

If task is complete, end with:
TOOL_CALLS:
[]"
            };

            let response = self.call_api(plan_prompt).await?;

            // Parse response to extract tool calls
            let (reasoning, tool_calls) = self.parse_response(&response)?;

            println!("{}", reasoning);
            println!();

            if tool_calls.is_empty() {
                self.output.info("Task complete");
                return Ok(());
            }

            // Step 2: Execute tool calls
            self.output.tool_header("Executing");
            for (i, call) in tool_calls.iter().enumerate() {
                self.output.list_item(i + 1, &format!("{} {}", call.tool, call.args.join(" ")));
            }
            println!();

            self.execute_tool_calls(tool_calls).await?;
            println!();
        }

        self.output.info(&format!("Reached max turns ({})", MAX_TURNS));
        Ok(())
    }

    fn parse_response(&self, response: &str) -> Result<(String, Vec<ToolCall>)> {
        if let Some(split_pos) = response.find("TOOL_CALLS:") {
            let reasoning = response[..split_pos].trim().to_string();
            let json_part = response[split_pos + 11..].trim();

            let cleaned_json = json_part
                .strip_prefix("```json")
                .or_else(|| json_part.strip_prefix("```"))
                .unwrap_or(json_part)
                .strip_suffix("```")
                .unwrap_or(json_part)
                .trim();

            let tool_calls: Vec<ToolCall> = serde_json::from_str(cleaned_json)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to parse tool calls: {}. JSON was:\n{}", e, cleaned_json);
                    vec![]
                });

            Ok((reasoning, tool_calls))
        } else {
            Ok((response.to_string(), vec![]))
        }
    }

    fn handle_error<'a>(&'a mut self, error: &'a str, failed_tool: &'a ToolCall, depth: u32) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
        Box::pin(async move {
            self.output.tool_header("Error Recovery");

            // Add error context to conversation
            self.add_user_message(&format!(
                "ERROR: The tool call '{}' with args {:?} failed with error: {}\n\n\
                Please analyze why this failed and try a different approach.",
                failed_tool.tool, failed_tool.args, error
            ));

            // Process with error context using unified flow
            self.process_with_retry(depth + 1).await?;

            Ok(())
        })
    }

    async fn process_with_retry(&mut self, depth: u32) -> Result<()> {
        const MAX_RETRIES: u32 = 2;

        if depth > MAX_RETRIES {
            self.output.error("Max retries reached");
            return Ok(());
        }

        // Think and plan with error context
        self.output.tool_header("Thinking");

        let plan_prompt = "\
Based on the conversation and error above, think step-by-step:
1. Why did the previous approach fail?
2. What should I try instead?
3. What tools do I need?

Then output tool calls as a JSON array.

Available tools:
- read <path>: Read a file
- write <path> <content>: Write a file
- edit <path> <search> <replace>: Edit a file
- bash <command>: Run shell command
- glob <pattern>: Find files
- grep <pattern> <path>: Search contents

Format:
[Your reasoning]

TOOL_CALLS:
[{\"tool\": \"read\", \"args\": [\"path\"], \"description\": \"what this does\"}]";

        let response = self.call_api(plan_prompt).await?;
        let (reasoning, tool_calls) = self.parse_response(&response)?;

        println!("{}", reasoning);
        println!();

        if tool_calls.is_empty() {
            self.output.info("No alternative approach found");
            return Ok(());
        }

        self.output.tool_header("Executing");
        for (i, call) in tool_calls.iter().enumerate() {
            self.output.list_item(i + 1, &format!("{} {}", call.tool, call.args.join(" ")));
        }
        println!();

        self.execute_tool_calls_with_retry(tool_calls, depth).await?;

        Ok(())
    }

    async fn call_api(&mut self, prompt: &str) -> Result<String> {
        // Add user message to history
        self.messages.push(Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request = AnthropicRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 8000,
            system: self.system_prompt.clone(),
            messages: self.messages.clone(),
        };

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let response_body: AnthropicResponse = response.json().await?;

        let assistant_message = response_body.content.first()
            .map(|c| c.text.clone())
            .unwrap_or_else(|| "No response".to_string());

        // Add assistant response to history
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: assistant_message.clone(),
        });

        Ok(assistant_message)
    }

    pub async fn execute_tool_calls(&mut self, tool_calls: Vec<ToolCall>) -> Result<()> {
        self.execute_tool_calls_with_retry(tool_calls, 0).await
    }

    async fn execute_tool_calls_with_retry(&mut self, tool_calls: Vec<ToolCall>, depth: u32) -> Result<()> {
        const MAX_RETRIES: u32 = 2;

        self.output.tool_header("Executing");

        let mut results = Vec::new();
        let mut had_error = false;
        let mut failed_call: Option<ToolCall> = None;
        let mut error_msg = String::new();

        for (i, call) in tool_calls.iter().enumerate() {
            self.output.info(&format!("\n[{}/{}] {}", i + 1, tool_calls.len(), call.description));

            match self.execute_single_tool(&call).await {
                Ok(result) => {
                    if !result.is_empty() {
                        println!("{}", result);
                    }
                    self.output.success("✓ Done");
                    results.push(format!("{}: {}", call.description, result));
                }
                Err(e) => {
                    let err_str = format!("{}", e);
                    self.output.error(&format!("✗ Error: {}", err_str));
                    results.push(format!("{}: Error - {}", call.description, err_str));

                    if depth < MAX_RETRIES {
                        had_error = true;
                        failed_call = Some(call.clone());
                        error_msg = err_str;
                        break;
                    }
                }
            }
        }

        println!();

        // Add tool execution results to conversation history
        if !results.is_empty() {
            let results_summary = format!("Tool execution results:\n{}", results.join("\n"));
            self.messages.push(Message {
                role: "user".to_string(),
                content: results_summary,
            });
        }

        // If error occurred and we haven't hit max retries, try error recovery
        if had_error {
            if let Some(call) = failed_call {
                println!();
                self.output.info(&format!("Retry attempt {}/{}", depth + 1, MAX_RETRIES));
                self.handle_error(&error_msg, &call, depth).await?;
            }
        } else {
            self.output.success("All tasks completed!");
        }

        Ok(())
    }

    async fn execute_single_tool(&self, call: &ToolCall) -> Result<String> {
        match call.tool.as_str() {
            "read" => {
                let path = call.args.get(0).map(|s| s.as_str()).unwrap_or("");
                let content = tools::read(path, None, None)?;
                Ok(content)
            }
            "write" => {
                let path = call.args.get(0).map(|s| s.as_str()).unwrap_or("");
                let content = call.args[1..].join(" ");
                tools::write(path, &content)?;
                Ok(format!("Wrote to {}", path))
            }
            "edit" => {
                let path = call.args.get(0).map(|s| s.as_str()).unwrap_or("");
                let search = call.args.get(1).map(|s| s.as_str()).unwrap_or("");
                let replace = call.args.get(2).map(|s| s.as_str()).unwrap_or("");
                let diff = tools::edit(path, search, replace, false)?;
                Ok(format!("Edited {}\n{}", path, diff))
            }
            "bash" => {
                let command = call.args.join(" ");
                let result = tools::bash(&command).await?;
                Ok(result.output)
            }
            "glob" => {
                let pattern = call.args.get(0).map(|s| s.as_str()).unwrap_or("");
                let matches = tools::glob(pattern, None)?;
                Ok(format!("Found {} files:\n{}", matches.len(), matches.join("\n")))
            }
            "grep" => {
                let pattern = call.args.get(0).map(|s| s.as_str()).unwrap_or("");
                let path = call.args.get(1).map(|s| s.as_str()).unwrap_or(".");
                let matches = tools::grep(pattern, path, false)?;
                let output = matches.iter()
                    .take(10)
                    .map(|m| format!("{}:{} {}", m.file, m.line_num, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(format!("Found {} matches:\n{}", matches.len(), output))
            }
            _ => Err(anyhow::anyhow!("Unknown tool: {}", call.tool)),
        }
    }
}
