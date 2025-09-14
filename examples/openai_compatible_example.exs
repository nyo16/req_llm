# OpenAI-Compatible Endpoint Example
#
# This demonstrates ReqLLM's compatible_with_openai provider, which works with:
# • LM Studio, Ollama, MLX, LocalAI, and any OpenAI-compatible server
# • Any model name (no validation restrictions)
# • Optional authentication for local/remote servers
#
# Start your server: mlx_lm.server --model qwen3-4b-instruct-2507-mlx --host 0.0.0.0 --port 1234

IO.puts("🚀 CompatibleWithOpenAI Provider Example")

# Create model - any name works!
model = ReqLLM.Model.from!("compatible_with_openai:qwen3-4b-instruct-2507-mlx")
IO.puts("✓ Model: #{model.provider}:#{model.model}")

# Test different model names
["llama-3.1-8b", "mistral-7b", "any-custom-name"]
|> Enum.each(fn name ->
  test_model = ReqLLM.Model.from!("compatible_with_openai:#{name}")
  IO.puts("✓ #{test_model.model}")
end)

IO.puts("\n📖 Usage Examples:")
IO.puts("""
# Basic usage:
{:ok, response} = ReqLLM.generate_text(
  model,
  "What day is it today?",
  temperature: 0.7,
  system_prompt: "Always answer in rhymes",
  provider_options: [base_url: "http://localhost:1234/v1"]
)

text = ReqLLM.Response.text(response)

# With authentication:
provider_options: [
  base_url: "https://your-server.com/v1",
  api_key: "your-key"
]

# Streaming:
{:ok, response} = ReqLLM.stream_text(model, "Tell me a story",
  provider_options: [base_url: "http://localhost:1234/v1"])

ReqLLM.Response.text_stream(response) |> Enum.each(&IO.write/1)
""")

IO.puts("✅ Ready to use with your OpenAI-compatible server!")