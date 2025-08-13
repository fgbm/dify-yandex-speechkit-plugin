import unittest
from unittest.mock import MagicMock, patch

from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from provider.yandex_speechkit import YandexSpeechkitProvider


class TestYandexSpeechkitProvider(unittest.TestCase):
    """Test cases for YandexSpeechkitProvider"""

    def setUp(self):
        """Set up test fixtures"""
        self.provider = YandexSpeechkitProvider()
        self.valid_credentials = {"api_key": "test-api-key"}

    def test_validate_credentials_missing_api_key(self):
        """Test validation with missing API key"""
        credentials = {}

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(credentials)

        self.assertIn("API key is required", str(context.exception))

    def test_validate_credentials_empty_api_key(self):
        """Test validation with empty API key"""
        credentials = {"api_key": ""}

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(credentials)

        self.assertIn("API key is required", str(context.exception))

    def test_validate_credentials_non_string_api_key(self):
        """Test validation with non-string API key"""
        credentials = {"api_key": 12345}

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(credentials)

        self.assertIn("API key is required", str(context.exception))

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_successful(self, mock_from_credentials):
        """Test successful credential validation"""
        # Mock successful TTS tool creation and invocation
        mock_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.message.text = "Speech synthesis completed successfully!"
        mock_tool.invoke.return_value = [mock_result]
        mock_from_credentials.return_value = mock_tool

        # Should not raise an exception
        try:
            self.provider._validate_credentials(self.valid_credentials)
        except ToolProviderCredentialValidationError:
            self.fail(
                "_validate_credentials raised ToolProviderCredentialValidationError unexpectedly!"
            )

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_api_error(self, mock_from_credentials):
        """Test credential validation with API error"""
        # Mock TTS tool that returns error
        mock_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.message.text = "Error: API error: 401 - Invalid API key"
        mock_tool.invoke.return_value = [mock_result]
        mock_from_credentials.return_value = mock_tool

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn("API validation failed", str(context.exception))

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_no_response(self, mock_from_credentials):
        """Test credential validation with no response"""
        # Mock TTS tool that returns empty result
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = []
        mock_from_credentials.return_value = mock_tool

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn("No response from Yandex SpeechKit API", str(context.exception))

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_unauthorized_error(self, mock_from_credentials):
        """Test credential validation with 401 error"""
        # Mock TTS tool that raises exception with 401 error
        mock_from_credentials.side_effect = Exception("HTTP 401 Unauthorized")

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn("Invalid API key - unauthorized access", str(context.exception))

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_forbidden_error(self, mock_from_credentials):
        """Test credential validation with 403 error"""
        # Mock TTS tool that raises exception with 403 error
        mock_from_credentials.side_effect = Exception("HTTP 403 Forbidden")

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn(
            "API key does not have required permissions", str(context.exception)
        )

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_timeout_error(self, mock_from_credentials):
        """Test credential validation with timeout error"""
        # Mock TTS tool that raises timeout exception
        mock_from_credentials.side_effect = Exception("Request timeout")

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn("API request timeout", str(context.exception))

    @patch("tools.text_to_speech.TextToSpeechTool.from_credentials")
    def test_validate_credentials_generic_error(self, mock_from_credentials):
        """Test credential validation with generic error"""
        # Mock TTS tool that raises generic exception
        mock_from_credentials.side_effect = Exception("Some generic error")

        with self.assertRaises(ToolProviderCredentialValidationError) as context:
            self.provider._validate_credentials(self.valid_credentials)

        self.assertIn(
            "Credential validation failed: Some generic error", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
