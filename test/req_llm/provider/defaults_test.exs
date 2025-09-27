defmodule ReqLLM.Provider.DefaultsTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Provider.Defaults
  alias ReqLLM.{Context, Message, Message.ContentPart, Model, StreamChunk}

  describe "encode_context_to_openai_format/2" do
    test "encodes text content correctly" do
      test_cases = [
        # Simple string content
        {%Message{role: :user, content: "Hello"}, "Hello"},
        # Single text part flattens to string
        {%Message{role: :user, content: [%ContentPart{type: :text, text: "Hello"}]}, "Hello"},
        # Multiple text parts stay as array
        {%Message{
           role: :user,
           content: [
             %ContentPart{type: :text, text: "Hello"},
             %ContentPart{type: :text, text: "World"}
           ]
         }, [%{type: "text", text: "Hello"}, %{type: "text", text: "World"}]}
      ]

      for {message, expected_content} <- test_cases do
        context = %Context{messages: [message]}
        result = Defaults.encode_context_to_openai_format(context, "gpt-4")

        assert result == %{messages: [%{role: "user", content: expected_content}]}
      end
    end

    test "encodes tool calls correctly" do
      # Test message-level tool_calls
      message_tool_calls = %Message{
        role: :assistant,
        content: [],
        tool_calls: [
          %{
            id: "call_123",
            type: "function",
            function: %{name: "get_weather", arguments: ~s({"city":"New York"})}
          }
        ]
      }

      # Test content-part level tool calls
      content_tool_calls = %Message{
        role: :assistant,
        content: [
          %ContentPart{
            type: :tool_call,
            tool_name: "get_weather",
            input: %{city: "New York"},
            tool_call_id: "call_123"
          }
        ]
      }

      expected_message_result = %{
        messages: [
          %{
            role: "assistant",
            content: [],
            tool_calls: [
              %{
                id: "call_123",
                type: "function",
                function: %{name: "get_weather", arguments: ~s({"city":"New York"})}
              }
            ]
          }
        ]
      }

      expected_content_result = %{
        messages: [
          %{
            role: "assistant",
            content: [
              %{
                id: "call_123",
                type: "function",
                function: %{name: "get_weather", arguments: ~s({"city":"New York"})}
              }
            ]
          }
        ]
      }

      assert Defaults.encode_context_to_openai_format(
               %Context{messages: [message_tool_calls]},
               "gpt-4"
             ) == expected_message_result

      assert Defaults.encode_context_to_openai_format(
               %Context{messages: [content_tool_calls]},
               "gpt-4"
             ) == expected_content_result
    end
  end

  describe "decode_response_body_openai_format/2" do
    setup do
      %{model: %Model{provider: :openai, model: "gpt-4"}}
    end

    test "decodes responses correctly", %{model: model} do
      test_cases = [
        # Basic text response
        {%{
           "id" => "chatcmpl-123",
           "model" => "gpt-4",
           "choices" => [
             %{"message" => %{"content" => "Hello there!"}, "finish_reason" => "stop"}
           ],
           "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}
         },
         fn result ->
           assert result.id == "chatcmpl-123"
           assert result.finish_reason == :stop
           assert result.usage == %{input_tokens: 10, output_tokens: 5, total_tokens: 15}
           assert result.message.content == [%ContentPart{type: :text, text: "Hello there!"}]
         end},

        # Tool call response
        {%{
           "id" => "chatcmpl-456",
           "choices" => [
             %{
               "message" => %{
                 "tool_calls" => [
                   %{
                     "id" => "call_123",
                     "type" => "function",
                     "function" => %{
                       "name" => "get_weather",
                       "arguments" => ~s({"city":"New York"})
                     }
                   }
                 ]
               },
               "finish_reason" => "tool_calls"
             }
           ]
         },
         fn result ->
           assert result.finish_reason == :tool_calls
           assert [tool_call_part] = result.message.content
           assert tool_call_part.type == :tool_call
           assert tool_call_part.tool_name == "get_weather"
           assert tool_call_part.input == %{"city" => "New York"}
           assert tool_call_part.tool_call_id == "call_123"
         end},

        # Missing fields handled gracefully  
        {%{"choices" => [%{"message" => %{"content" => "Hello"}}]},
         fn result ->
           assert result.id == "unknown"
           assert result.model == "gpt-4"
           assert result.usage == %{input_tokens: 0, output_tokens: 0, total_tokens: 0}
           assert result.finish_reason == nil
         end}
      ]

      for {response_data, assertion_fn} <- test_cases do
        {:ok, result} = Defaults.decode_response_body_openai_format(response_data, model)
        assertion_fn.(result)
      end
    end
  end

  describe "default_decode_sse_event/2" do
    setup do
      %{model: %Model{provider: :openai, model: "gpt-4"}}
    end

    test "decodes streaming events correctly", %{model: model} do
      # Content delta
      content_event = %{data: %{"choices" => [%{"delta" => %{"content" => "Hello"}}]}}

      assert Defaults.default_decode_sse_event(content_event, model) == [
               %StreamChunk{type: :content, text: "Hello"}
             ]

      # Tool call delta with valid JSON
      tool_event = %{
        data: %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "id" => "call_123",
                    "type" => "function",
                    "function" => %{
                      "name" => "get_weather",
                      "arguments" => ~s({"city":"New York"})
                    }
                  }
                ]
              }
            }
          ]
        }
      }

      [chunk] = Defaults.default_decode_sse_event(tool_event, model)
      assert chunk.type == :tool_call
      assert chunk.name == "get_weather"
      assert chunk.arguments == %{"city" => "New York"}
      assert chunk.metadata == %{id: "call_123"}
    end

    test "handles edge cases gracefully", %{model: model} do
      # Empty/invalid events
      assert Defaults.default_decode_sse_event(%{data: %{}}, model) == []
      assert Defaults.default_decode_sse_event(%{}, model) == []
      assert Defaults.default_decode_sse_event("invalid", model) == []

      # Tool call with invalid JSON - should fallback to empty map
      invalid_json_event = %{
        data: %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "id" => "call_123",
                    "type" => "function",
                    "function" => %{"name" => "get_weather", "arguments" => "invalid json"}
                  }
                ]
              }
            }
          ]
        }
      }

      [chunk] = Defaults.default_decode_sse_event(invalid_json_event, model)
      assert chunk.type == :tool_call
      assert chunk.arguments == %{}
    end
  end
end
