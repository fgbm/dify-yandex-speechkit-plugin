import os
import unittest
from unittest.mock import MagicMock, patch

from tools.text_to_speech import TextToSpeechTool


class TestTextToSpeechTool(unittest.TestCase):
    """Test cases for TextToSpeechTool"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv("YANDEX_API_KEY", "test-api-key")
        self.credentials = {"api_key": self.api_key}
        self.tool = TextToSpeechTool.from_credentials(self.credentials)

    def test_validate_parameters_success(self):
        """Test successful parameter validation"""
        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        result = self.tool._validate_parameters(test_params)

        self.assertEqual(result["text"], "Тестовый текст")
        self.assertEqual(result["voice"], "marina")
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["speed"], "1.0")
        self.assertEqual(result["format"], "mp3")

    def test_validate_parameters_empty_text(self):
        """Test validation with empty text"""
        test_params = {
            "text": "",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Text content is required", str(context.exception))

    def test_validate_parameters_long_text(self):
        """Test validation with too long text"""
        test_params = {
            "text": "a" * 5001,  # Too long
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Text is too long", str(context.exception))

    def test_validate_parameters_invalid_voice(self):
        """Test validation with invalid voice"""
        test_params = {
            "text": "Тестовый текст",
            "voice": "invalid_voice",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Invalid voice", str(context.exception))

    def test_validate_parameters_invalid_emotion(self):
        """Test validation with invalid emotion"""
        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "invalid_emotion",
            "speed": 1.0,
            "format": "mp3",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Invalid emotion", str(context.exception))

    def test_validate_parameters_invalid_speed(self):
        """Test validation with invalid speed"""
        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 5.0,  # Too high
            "format": "mp3",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Speed must be between 0.1 and 3.0", str(context.exception))

    def test_validate_parameters_invalid_format(self):
        """Test validation with invalid format"""
        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "invalid_format",
        }

        with self.assertRaises(ValueError) as context:
            self.tool._validate_parameters(test_params)

        self.assertIn("Invalid format", str(context.exception))

    @patch("requests.post")
    def test_successful_synthesis(self, mock_post):
        """Test successful text-to-speech synthesis"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_post.return_value = mock_response

        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        # Invoke tool
        results = list(self.tool._invoke(test_params))

        # Verify results
        self.assertEqual(len(results), 2)  # Text message + blob message

        # Check text message
        text_result = results[0]
        self.assertTrue(hasattr(text_result, "message"))
        self.assertIn(
            "Speech synthesis completed successfully", text_result.message.text
        )

        # Check blob message
        blob_result = results[1]
        self.assertTrue(hasattr(blob_result, "message"))
        self.assertEqual(blob_result.message.blob, b"fake_audio_data")

    @patch("requests.post")
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}
        mock_post.return_value = mock_response

        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("API error: 401", result.message.text)

    @patch("requests.post")
    def test_empty_response_handling(self, mock_post):
        """Test handling of empty API response"""
        # Mock API response with empty content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b""
        mock_post.return_value = mock_response

        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("Empty response from TTS service", result.message.text)

    def test_missing_api_key(self):
        """Test handling of missing API key"""
        tool_without_key = TextToSpeechTool.from_credentials({})
        test_params = {
            "text": "Тестовый текст",
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        results = list(tool_without_key._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.message.text, "Error: API key is required")

    def test_parameter_validation_error(self):
        """Test handling of parameter validation errors"""
        test_params = {
            "text": "",  # Invalid empty text
            "voice": "marina",
            "emotion": "neutral",
            "speed": 1.0,
            "format": "mp3",
        }

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("Parameter error", result.message.text)

    def test_voice_options(self):
        """Test different voice options"""
        voices = [
            "marina",
            "alena",
            "filipp",
            "jane",
            "omazh",
            "ermil",
            "zahar",
            "madi_ru",
        ]

        for voice in voices:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_audio_data"
                mock_post.return_value = mock_response

                test_params = {
                    "text": "Тестовый текст",
                    "voice": voice,
                    "emotion": "neutral",
                    "speed": 1.0,
                    "format": "mp3",
                }

                list(self.tool._invoke(test_params))

                # Verify API was called with correct voice
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["data"]["voice"], voice)

    def test_emotion_options(self):
        """Test different emotion options"""
        emotions = ["neutral", "good", "evil", "friendly", "whisper"]

        for emotion in emotions:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_audio_data"
                mock_post.return_value = mock_response

                test_params = {
                    "text": "Тестовый текст",
                    "voice": "marina",
                    "emotion": emotion,
                    "speed": 1.0,
                    "format": "mp3",
                }

                list(self.tool._invoke(test_params))

                # Verify API was called with correct emotion
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["data"]["emotion"], emotion)

    def test_format_options(self):
        """Test different format options"""
        formats = ["mp3", "wav", "opus"]

        for format_type in formats:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_audio_data"
                mock_post.return_value = mock_response

                test_params = {
                    "text": "Тестовый текст",
                    "voice": "marina",
                    "emotion": "neutral",
                    "speed": 1.0,
                    "format": format_type,
                }

                results = list(self.tool._invoke(test_params))

                # Verify API was called with correct format
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["data"]["format"], format_type)

                # Verify blob message exists (meta might not be accessible in test)
                blob_result = results[1]
                self.assertTrue(hasattr(blob_result, "message"))

    def test_speed_options(self):
        """Test different speed options"""
        speeds = [0.1, 0.5, 1.0, 2.0, 3.0]

        for speed in speeds:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_audio_data"
                mock_post.return_value = mock_response

                test_params = {
                    "text": "Тестовый текст",
                    "voice": "marina",
                    "emotion": "neutral",
                    "speed": speed,
                    "format": "mp3",
                }

                list(self.tool._invoke(test_params))

                # Verify API was called with correct speed
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["data"]["speed"], str(speed))


if __name__ == "__main__":
    unittest.main()
