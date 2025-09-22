defmodule ReqLLM.Coverage.VLLM.CoreTest do
  @moduledoc """
  Core vLLM API feature coverage tests using simple fixtures.

  Run with LIVE=true to test against live API and record fixtures.
  Otherwise uses fixtures for fast, reliable testing.
  """

  use ReqLLM.ProviderTest.Core,
    provider: :vllm,
    model: "vllm:llama-3.1-8b-instant"

  # vLLM-specific tests
  test "extra sampling parameters" do
    {:ok, response} =
      ReqLLM.generate_text(
        "vllm:llama-3.1-8b-instant",
        "What is 2+2?",
        temperature: 0.0,
        max_tokens: 10,
        provider_options: [
          repetition_penalty: 1.1,
          min_p: 0.05
        ],
        fixture: "extra_sampling"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil
    assert text =~ "4"
  end

  test "guided json generation" do
    schema = %{
      "type" => "object",
      "properties" => %{
        "answer" => %{"type" => "integer"},
        "reasoning" => %{"type" => "string"}
      },
      "required" => ["answer", "reasoning"]
    }

    {:ok, response} =
      ReqLLM.generate_text(
        "vllm:llama-3.1-8b-instant",
        "What is 5 + 3? Provide both the answer and your reasoning.",
        temperature: 0.0,
        max_tokens: 100,
        provider_options: [
          guided_json: schema
        ],
        fixture: "guided_json"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil

    # Should be valid JSON matching the schema
    {:ok, parsed} = Jason.decode(text)
    assert is_integer(parsed["answer"])
    assert is_binary(parsed["reasoning"])
  end

  test "guided choice generation" do
    {:ok, response} =
      ReqLLM.generate_text(
        "vllm:llama-3.1-8b-instant",
        "Is the sky blue? Answer with yes, no, or maybe.",
        temperature: 0.0,
        max_tokens: 5,
        provider_options: [
          guided_choice: ["yes", "no", "maybe"]
        ],
        fixture: "guided_choice"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil

    # Should be one of the guided choices
    normalized_text = String.trim(String.downcase(text))
    assert normalized_text in ["yes", "no", "maybe"]
  end

  test "beam search decoding" do
    {:ok, response} =
      ReqLLM.generate_text(
        "vllm:llama-3.1-8b-instant",
        "Tell me a brief fact about science",
        temperature: 0.0,
        max_tokens: 20,
        provider_options: [
          use_beam_search: true,
          best_of: 3
        ],
        fixture: "beam_search"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil
  end
end
