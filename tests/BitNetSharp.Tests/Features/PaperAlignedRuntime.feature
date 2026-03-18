Feature: Paper-aligned runtime use cases
  The test suite should express the documented use cases for the paper-aligned BitNet runtime.

  Scenario: Inspect next-token predictions for a prompt
    Given the default paper-aligned BitNet model
    When I generate a response for the prompt "how are you hosted"
    Then the response should list top next-token predictions
    And the response should include generated tokens
    And the diagnostics should describe the decoder-only transformer

  Scenario: Inspect ternary weight distribution across the seeded transformer
    Given the default paper-aligned BitNet model
    When I inspect the ternary weight distribution
    Then the ternary distribution should include negative zero and positive counts
    And the ternary distribution should include both negative and positive weights

  Scenario: Build the agent host for the paper-aligned model
    Given the default paper-aligned BitNet model
    When I build the agent host
    Then the host summary should describe the BitNet agent registration
