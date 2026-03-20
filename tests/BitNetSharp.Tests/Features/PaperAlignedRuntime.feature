Feature: Hosted model runtime use cases
  The test suite should express measurable and comparable hosted-model use cases.

  Scenario Outline: Generate a response for a prompt
    Given the hosted model named "<model>"
    When I generate a response for the prompt "how are you hosted"
    Then the response text should be non-empty
    And the response should identify the selected model
    Examples:
      | model             |
      | bitnet-b1.58-sharp |
      | traditional-local  |

  Scenario Outline: Stream a response for a prompt
    Given the hosted model named "<model>"
    When I stream a response for the prompt "how are you hosted"
    Then the stream should include at least one update
    And each stream update should identify the selected model
    Examples:
      | model             |
      | bitnet-b1.58-sharp |
      | traditional-local  |

  Scenario Outline: Build the agent host for the selected model
    Given the hosted model named "<model>"
    When I build the agent host
    Then the host summary should describe the selected model registration
    Examples:
      | model             |
      | bitnet-b1.58-sharp |
      | traditional-local  |

  Scenario: Inspect the paper-aligned transformer description
    Given the hosted model named "bitnet-b1.58-sharp"
    When I inspect the selected model description
    Then the model description should enumerate the paper-aligned transformer topology

  Scenario: Run the paper-alignment audit for the canonical BitNet model
    Given the hosted model named "bitnet-b1.58-sharp"
    When I run the paper-alignment audit
    Then the paper-alignment architecture checks should all pass
    And the paper-alignment audit should verify repository runtime coverage

  Scenario Outline: Train the selected model on the default dataset
    Given the hosted model named "<model>"
    When I train the selected model on the default dataset
    Then the training run should complete over the default dataset
    Examples:
      | model            |
      | bitnet-b1.58-sharp |
      | traditional-local |
