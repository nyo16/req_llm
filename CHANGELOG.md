# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-rc.3] - 2025-01-22

### Added
- New Mix tasks for local testing and exploration:
  - generate_text, generate_object (structured output), and stream_object
  - All tasks support --log-level and --debug-dir for easier debugging; stream_text gains debug logging
- New providers: Alibaba (China) and Z.AI Coding Plan
- Google provider:
  - File content parts support (binary uploads via base64) for improved multimodal inputs
  - Added Gemini Embedding 001 support
- Model capability discovery and validation to catch unsupported features early (e.g., streaming, tools, structured output, embeddings)
- Streaming utilities to capture raw SSE chunks and save streaming fixtures
- Schema validation utilities for structured outputs with clearer, actionable errors

### Enhanced
- Major provider refactor to a unified, codec-based architecture
  - More consistent request/response handling across providers and improved alignment with OpenAI semantics
- Streaming reliability and performance improvements (better SSE parsing and handling)
- Centralized model metadata handling for more accurate capabilities and configuration
- Error handling and logging across the library for clearer diagnostics and easier troubleshooting
- Embedding flow robustness and coverage

### Fixed
- More informative errors on invalid/partial provider responses and schema mismatches
- Stability improvements in streaming and fixture handling across providers

### Changed
- jido_keys is now a required dependency (installed transitively; no code changes expected for most users)
- Logging warnings standardized to Logger.warning

### Internal
- Testing infrastructure overhaul:
  - New timing-aware LLMFixture system, richer streaming/object/tool-calling fixtures, and broader provider coverage
  - Fake API key support for safer, more reliable test runs

### Notes
- No public API-breaking changes are expected; upgrades should be seamless for most users

## [1.0.0-rc.2] - 2025-01-15

### Added
- Model metadata guide with comprehensive documentation for managing AI model information
- Local patching system for model synchronization, allowing custom model metadata overrides
- `.env.example` file to guide API key setup and configuration
- GitHub configuration files for automated dependency management and issue tracking
- Test coverage reporting with ExCoveralls integration
- Centralized `ReqLLM.Keys` module for unified API key management with clear precedence order

### Fixed
- **BREAKING**: Bang methods (`generate_text!/3`, `stream_text!/3`, `generate_object!/4`) now return naked values instead of `{:ok, result}` tuples ([#9](https://github.com/agentjido/req_llm/pull/9))
- OpenAI o1 and o3 model parameter translation - automatic conversion of `max_tokens` to `max_completion_tokens` and removal of unsupported `temperature` parameter ([#8](https://github.com/agentjido/req_llm/issues/8), [#11](https://github.com/agentjido/req_llm/pull/11))
- Mix task for streaming text updated to work with new bang method patterns
- Embedding method documentation updated from `generate_embeddings/2` to `embed_many/2`

### Enhanced
- Provider architecture with new `translate_options/3` callback for model-specific parameter handling
- API key management system with centralized `ReqLLM.Keys` module supporting multiple source precedence
- Documentation across README.md, guides, and usage-rules.md for improved clarity and accuracy
- GitHub workflow and dependency management with Dependabot automation
- Response decoder modules streamlined by removing unused Model aliases
- Mix.exs configuration with improved Dialyzer setup and dependency organization

### Technical Improvements
- Added validation for conflicting provider parameters with `validate_mutex!/3`
- Enhanced error handling for unsupported parameter translations
- Comprehensive test coverage for new translation functionality
- Model synchronization with local patch merge capabilities
- Improved documentation structure and formatting across all guides

### Infrastructure
- Weekly automated dependency updates via Dependabot
- Standardized pull request and issue templates
- Enhanced CI workflow with streamlined checks
- Test coverage configuration and reporting setup

## [1.0.0-rc.1] - 2025-01-13

### Added
- First public release candidate
- Composable plugin architecture built on Req
- Support for 45+ providers and 665+ models via models.dev sync
- Typed data structures for all API interactions
- Dual API layers: low-level Req plugin and high-level helpers
- Built-in streaming support with typed StreamChunk responses
- Automatic usage and cost tracking
- Anthropic and OpenAI provider implementations
- Context Codec protocol for provider wire format conversion
- JidoKeys integration for secure API key management
- Comprehensive test matrix with fixture and live testing support
- Tool calling capabilities
- Embeddings generation support (OpenAI)
- Structured data generation with schema validation
- Extensive documentation and guides

### Features
- `ReqLLM.generate_text/3` and `generate_text!/3` for text generation
- `ReqLLM.stream_text/3` and `stream_text!/3` for streaming responses  
- `ReqLLM.generate_object/4` and `generate_object!/4` for structured output
- `ReqLLM.generate_embeddings/3` for vector embeddings
- `ReqLLM.run/3` for low-level Req plugin integration
- Provider-agnostic model specification with "provider:model" syntax
- Automatic model metadata loading and cost calculation
- Tool definition and execution framework
- Message and content part builders
- Usage statistics and cost tracking on all responses

### Technical
- Elixir ~> 1.15 compatibility
- OTP 24+ support  
- Apache-2.0 license
- Comprehensive documentation with HexDocs
- Quality tooling with Dialyzer, Credo, and formatter
- LiveFixture testing framework for API mocking

[1.0.0-rc.3]: https://github.com/agentjido/req_llm/releases/tag/v1.0.0-rc.3
[1.0.0-rc.2]: https://github.com/agentjido/req_llm/releases/tag/v1.0.0-rc.2
[1.0.0-rc.1]: https://github.com/agentjido/req_llm/releases/tag/v1.0.0-rc.1
