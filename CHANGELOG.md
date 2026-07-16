# CHANGELOG

<!-- version list -->

## v2.8.0 (2026-07-16)

### Bug Fixes

- **provider**: Cap 429 retry delays and remove dead fallback
  ([`a57dc0c`](https://github.com/nachollorca/lmdk/commit/a57dc0c3478a6c5312a11ce93bd24aecd6775d02))

### Features

- Implement exponential backoff and retry logic for requests
  ([`d91cc11`](https://github.com/nachollorca/lmdk/commit/d91cc117d945f83e28b4d73540e9b72df3e01755))


## v2.7.0 (2026-07-07)

### Features

- Add calling_service parameter for enhanced telemetry tracking
  ([`8a5cf12`](https://github.com/nachollorca/lmdk/commit/8a5cf12eda1ed1c3fdc74132ed003a7cab151bee))


## v2.6.0 (2026-06-23)

### Bug Fixes

- **tests**: Satisfy ty checks for completion helpers
  ([`f9b94e5`](https://github.com/nachollorca/lmdk/commit/f9b94e597312d530294e060f365d6a6c3769e345))

### Continuous Integration

- **just**: Pass --frozen to uv run commands
  ([`9806785`](https://github.com/nachollorca/lmdk/commit/98067857b3c7372d91390484c734c56939110722))

### Refactoring

- **ci**: Pass filenames from pre-commit hooks to just recipes
  ([`9895913`](https://github.com/nachollorca/lmdk/commit/98959137680e0607c640affb38184791499df727))

### Testing

- Enforce typing, formatting, and complexity across all Python files
  ([`f819fa0`](https://github.com/nachollorca/lmdk/commit/f819fa0cca1ad42d4667167c05ba9921ae7fbfef))


## v2.5.1 (2026-06-18)

### Bug Fixes

- **anthropic**: Now anthropic rejects sampling params like temperature, so we have to remove our 0
  default
  ([`e7e3883`](https://github.com/nachollorca/lmdk/commit/e7e3883f95e009d29ea985c8f7b39221df8fd013))

### Documentation

- Correct command in readme, add short general agent context
  ([`c12c3e8`](https://github.com/nachollorca/lmdk/commit/c12c3e8755a9589641edebe6fdf0fbdbad73722c))


## v2.5.0 (2026-06-16)

### Features

- Add strict option to output schema in Mistral payload
  ([`0fe6f03`](https://github.com/nachollorca/lmdk/commit/0fe6f03ccc8f2b15c4d3bebe8e1219cd6ef869bc))

### Refactoring

- Implement JSON-Schema normalization for strict structured output
  ([`30895b9`](https://github.com/nachollorca/lmdk/commit/30895b9cce149ed4e779ab77a39972fa60a80d5a))


## v2.4.0 (2026-06-12)

### Documentation

- **readme**: Add a reference to chatlas
  ([`2b3162a`](https://github.com/nachollorca/lmdk/commit/2b3162a21a99641f4d839ad885c70fbf7e5ea453))

### Features

- **render_template**: Make the stripping of curly braces optional
  ([`6bb0c0e`](https://github.com/nachollorca/lmdk/commit/6bb0c0e7766bc6c96c3c0d9d7b43231802d27d66))


## v2.3.0 (2026-05-29)

### Features

- **provider**: Made a generic local LM provider for OpenAI completions endpoint
  ([`8bf17fa`](https://github.com/nachollorca/lmdk/commit/8bf17fa00c6204d0d342e28885e89b1b5b374f0a))


## v2.2.0 (2026-05-28)

### Features

- **llamacpp**: Implement structured output
  ([`139f8c2`](https://github.com/nachollorca/lmdk/commit/139f8c2bd451e73f535b70af58fbf0b84899eb96))


## v2.1.1 (2026-05-21)

### Bug Fixes

- **batch**: Allow threads to carry context so we can observe the requests / responses
  ([`0893b09`](https://github.com/nachollorca/lmdk/commit/0893b09c2417961c9d38c8079b8ffea9f7f19777))


## v2.1.0 (2026-05-20)

### Features

- **vertex**: Add multi-region support
  ([`e815582`](https://github.com/nachollorca/lmdk/commit/e8155828c9ecf2f5990e486744db4d157fcd0e42))


## v2.0.0 (2026-05-19)

### Features

- Add first iteration of the Request / Response observer
  ([`39f3cd3`](https://github.com/nachollorca/lmdk/commit/39f3cd3f18301370e49c479933e5a0e798ce064a))

- **datatypes**: Separate CompletionBatch instead of using CompletionResponse.from_list
  ([`a7e1069`](https://github.com/nachollorca/lmdk/commit/a7e106904aaa07411696270941c36af6b129f0b6))

- **listener**: Implement observe to catch request/responses, make a listener to keep core lean
  ([`df3ab31`](https://github.com/nachollorca/lmdk/commit/df3ab31dd03969655a09d3df2077be7ba16e97ea))

### Breaking Changes

- **datatypes**: CompletionResponse.from_list does not exist anymore


## v1.8.0 (2026-05-07)

### Bug Fixes

- **telemetry**: Have parsed and output attributes NOT nested
  ([`6d80bde`](https://github.com/nachollorca/lmdk/commit/6d80bded1bdf46e73f50cffb9941a0edf839a5ab))

### Documentation

- Document optional telemetry feature
  ([`83a871d`](https://github.com/nachollorca/lmdk/commit/83a871dd530677a816293cd8d5f66b5b135e0ab0))

- **telemetry**: Remove extra comments
  ([`5e74ab6`](https://github.com/nachollorca/lmdk/commit/5e74ab6a155ff64d941a2a196c303b1b7114683d))

### Testing

- **telemetry**: Adjust tests for the not nested output/parsed
  ([`ba1d611`](https://github.com/nachollorca/lmdk/commit/ba1d611130e948b3baffc53517b804001fe7496e))


## v1.7.0 (2026-05-06)

### Continuous Integration

- Remove the otel exporter dep
  ([`7c1086f`](https://github.com/nachollorca/lmdk/commit/7c1086f7be8a6194be9a5d41356fb4f420eb8c7b))

### Documentation

- **readme**: Add telemetry exporter info
  ([`b2d388e`](https://github.com/nachollorca/lmdk/commit/b2d388e8e0faee94459800a25b367f426964445e))

### Refactoring

- **telemetry**: Make it leaner
  ([`dd242d1`](https://github.com/nachollorca/lmdk/commit/dd242d1f3a25f42ae918df00f0bc4d9fc3fa0106))


## v1.6.1 (2026-05-05)

### Bug Fixes

- **openai**: Generation kwargs
  ([`7090c7f`](https://github.com/nachollorca/lmdk/commit/7090c7fe0024c28182c0b4f436ce0b8299da250d))


## v1.6.0 (2026-05-03)

### Continuous Integration

- Remove duplicated arguments
  ([`9023803`](https://github.com/nachollorca/lmdk/commit/9023803caeec68e180b73ead21fdd9a9ecf8f679))

- Upgrade template to mold
  ([`0affd87`](https://github.com/nachollorca/lmdk/commit/0affd87cc20f89dac6b3389cd158831dd443b224))

### Features

- **core**: Allow users to get the CompletionRequest (including prompt) when calling the LMs
  ([`dc499ed`](https://github.com/nachollorca/lmdk/commit/dc499ed82e732a3861396f415b1bc2d0ab52d6fa))


## v1.5.0 (2026-04-13)

### Features

- Implement llamacpp provider
  ([`2021614`](https://github.com/nachollorca/lmdk/commit/2021614ed0ab50be5135023ad941e414269796eb))

### Testing

- Implement llamacpp tests
  ([`59ae4ec`](https://github.com/nachollorca/lmdk/commit/59ae4ec16f633212a0a72621f166c86a54071189))


## v1.4.0 (2026-04-02)

### Features

- Add anthropic provider
  ([`8939815`](https://github.com/nachollorca/lmdk/commit/8939815b5ada84f819ae0b7d4f7828dd74d2518a))


## v1.3.1 (2026-03-27)

### Bug Fixes

- Use global as default location for vertex
  ([`1acdf6a`](https://github.com/nachollorca/lmdk/commit/1acdf6a484caa7114dcfd85ae027615e0c9b9de3))


## v1.3.0 (2026-03-24)

### Code Style

- Rename PermissionError to APIPermission error to avoid overshadowing python's
  ([`745ca46`](https://github.com/nachollorca/lmdk/commit/745ca469fff45a5780bd34c3c1f37b5c74df068f))

### Features

- Allow for multiple environmental variables
  ([`0130b43`](https://github.com/nachollorca/lmdk/commit/0130b432902a43f0cf4107f8926a0a76cb650401))

- Overload required_env to allow for a single string or a tuple of env var names
  ([`7d86787`](https://github.com/nachollorca/lmdk/commit/7d8678751803ebc797e1c09e71574a0d4ac7cd1b))

### Refactoring

- Overload required_env to allow for a string or tuple of strings
  ([`56e0a88`](https://github.com/nachollorca/lmdk/commit/56e0a888d062f4d7b6d2301dacc5840dbbdb7915))


## v1.2.1 (2026-03-22)

### Bug Fixes

- Ensure vertex can call global endpoints
  ([`a35f907`](https://github.com/nachollorca/lmdk/commit/a35f90773268dbc1f61432aecbbdf021b6052307))


## v1.2.0 (2026-03-22)

### Continuous Integration

- Add meta to toml so that it shows up in PyPi
  ([`e918c1c`](https://github.com/nachollorca/lmdk/commit/e918c1c7d4d86ebedbaf553d8840982b3d02c9ef))

### Documentation

- Make an example of template rendering and expose the function in main init
  ([`11d0cc0`](https://github.com/nachollorca/lmdk/commit/11d0cc0b546a21bb8acf1d0edbacf194f23c5a81))

### Features

- Add jinja template rendering
  ([`1aba594`](https://github.com/nachollorca/lmdk/commit/1aba5944213f8e127c3a1ef38f37d8383c1f06d7))

### Refactoring

- Rename messages to prompt
  ([`c4f04e1`](https://github.com/nachollorca/lmdk/commit/c4f04e1d23627ac3888c79a6da831d85c3a013fb))


## v1.1.2 (2026-03-21)

### Bug Fixes

- Another dummy commit to check if the CD worklow works
  ([`2012f41`](https://github.com/nachollorca/lmdk/commit/2012f41f1ea4ee6e1b89870cc492777ada679840))


## v1.1.1 (2026-03-21)

### Bug Fixes

- Dummy commit to check if the CD worklow works
  ([`cc48cd1`](https://github.com/nachollorca/lmdk/commit/cc48cd16a16758cc50c666d6bdad2e7950a9ecb3))


## v1.1.0 (2026-03-21)

### Code Style

- Add some better type hinting
  ([`4bfd2ad`](https://github.com/nachollorca/lmdk/commit/4bfd2ad7b6aee2281f63ff4964946b789529de18))

### Continuous Integration

- Add token to fix semantic release
  ([`8e75a2e`](https://github.com/nachollorca/lmdk/commit/8e75a2e8ebb8ba8d3e1376e87ef36f5cb8bc1668))

- Make workflows more efficient, ignore verbi-pypi
  ([`8730f76`](https://github.com/nachollorca/lmdk/commit/8730f7610ae4055d2e9e1bb9fbab909b8049bb6e))

- Try to fix the release workflow
  ([`d39c6d7`](https://github.com/nachollorca/lmdk/commit/d39c6d7ba4eb641d2d6ed8ca792b62d42f3ab40a))

- Upgrade python-semantic-release and publish-action versions
  ([`ec6d890`](https://github.com/nachollorca/lmdk/commit/ec6d8909ed3d2fbb1b852fdb9134e5cbf0725c4c))

### Documentation

- Extend readme
  ([`658d871`](https://github.com/nachollorca/lmdk/commit/658d871473acc513c4a139cae7d23538ac3483b6))

- Extend readme and add license
  ([`cd87649`](https://github.com/nachollorca/lmdk/commit/cd87649460a8f96cc3e20be086edf7b9d234e1ab))

- Little readme change
  ([`e635b5f`](https://github.com/nachollorca/lmdk/commit/e635b5fadc071ca224fd0e52bc43acb705ee03ed))

- Make a better docstring for the Provider class
  ([`19b4d9b`](https://github.com/nachollorca/lmdk/commit/19b4d9bb4f9b0d0d00e91b688f82cba91f9e4ac4))

- Make expanders in the usage
  ([`9e85aef`](https://github.com/nachollorca/lmdk/commit/9e85aef11bc5ccda00c1e592524e9e9d65703545))

- TOC newline
  ([`7a53374`](https://github.com/nachollorca/lmdk/commit/7a533746a0e7b3f2c8afbba791416498f91a1ee1))

### Features

- Add example file, unify exception handling and solve typing quirks
  ([`cf9f0e0`](https://github.com/nachollorca/lmdk/commit/cf9f0e03ce885342c9941aa4ae1a09399f5a4d62))

- Add more errors
  ([`2dab995`](https://github.com/nachollorca/lmdk/commit/2dab995829ebac66ef3bbc4b90c3120a0ce5bbfa))

- Add structured output and stream to mistral provider
  ([`b90d0f4`](https://github.com/nachollorca/lmdk/commit/b90d0f40b34099389b930970089b77796771c76d))

- Add vertex provider
  ([`72dcd45`](https://github.com/nachollorca/lmdk/commit/72dcd4514a9d27d7e149400c02ed3da71fe82293))

- Add vertex provider
  ([`6715367`](https://github.com/nachollorca/lmdk/commit/6715367cc74b11ef93abd38a47ac15c491a86adb))

- Make example script runnable
  ([`544a3b3`](https://github.com/nachollorca/lmdk/commit/544a3b3bcea49c060773ad57ea50f7a1ba31b8d9))

### Refactoring

- From lmtk to lmdk
  ([`2dec044`](https://github.com/nachollorca/lmdk/commit/2dec0442c3ac2cd225764f26fd40c8dd37b6ea5d))

- Move RawResponse to datatypes
  ([`ed64469`](https://github.com/nachollorca/lmdk/commit/ed64469430c11dfb273e7123cf1d45d31f0305d7))

- Remove duplication in payload building
  ([`164bb9b`](https://github.com/nachollorca/lmdk/commit/164bb9baa145f24c8df5e4fd582295755d6c168a))

- Rename get_response to complete and get_response_batch to complete_batch
  ([`d8665f3`](https://github.com/nachollorca/lmdk/commit/d8665f3ec223471f9728473d79ba1fd0c7a1f3b7))


## v1.0.0 (2026-03-17)

- Initial Release
