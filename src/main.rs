use inquire::Text;
use forge::{Output, Agent};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let output = Output::new();
    output.info("Forging now...\n");

    let api_key = env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
        output.error("ANTHROPIC_API_KEY not found in environment");
        output.info("Set it with: export ANTHROPIC_API_KEY=your_key");
        output.info("Or create a .env file");
        std::process::exit(1);
    });

    let mut agent = Agent::new(api_key);

    loop {
        let input = match Text::new("you>").prompt() {
            Ok(i) => i,
            Err(_) => break,
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            output.success("Goodbye!");
            break;
        }

        println!();

        // Add user message to conversation history
        agent.add_user_message(input);

        // Process the request with full context
        if let Err(e) = agent.process().await {
            output.error(&format!("Error: {}", e));
        }

        println!();
    }

    Ok(())
}
