import os
import struct
import unittest
import wave
from io import BytesIO
from unittest.mock import MagicMock, patch

from tools.speech_to_text import SpeechToTextTool


class TestSpeechToTextTool(unittest.TestCase):
    """Test cases for SpeechToTextTool"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv("YANDEX_API_KEY", "test-api-key")
        self.credentials = {"api_key": self.api_key}
        self.tool = SpeechToTextTool.from_credentials(self.credentials)

    def create_test_wav(self, duration=1.0, sample_rate=16000, frequency=440):
        """Create a test WAV file with a sine wave"""
        frames = int(duration * sample_rate)
        wav_buffer = BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Generate sine wave
            for i in range(frames):
                value = int(
                    32767
                    * 0.3
                    * (i % (sample_rate // frequency))
                    / (sample_rate // frequency)
                )
                wav_file.writeframes(struct.pack("<h", value))

        wav_buffer.seek(0)
        return wav_buffer.getvalue()

    def test_convert_wav_to_pcm(self):
        """Test WAV to PCM conversion"""
        wav_data = self.create_test_wav()
        pcm_data, sample_rate = self.tool._convert_wav_to_pcm(wav_data)

        self.assertIsInstance(pcm_data, bytes)
        self.assertEqual(sample_rate, 16000)
        self.assertGreater(len(pcm_data), 0)

    def test_convert_audio_formats(self):
        """Test audio format conversion"""
        # Test with WAV data
        wav_data = self.create_test_wav()

        try:
            # This requires pydub to be installed
            converted_wav = self.tool._convert_audio_to_wav(wav_data)
            self.assertIsInstance(converted_wav, bytes)
            self.assertGreater(len(converted_wav), 0)
        except ValueError as e:
            if "Audio conversion library not available" in str(e):
                self.skipTest("pydub not available for testing")
            else:
                raise

    @patch("requests.post")
    def test_successful_recognition(self, mock_post):
        """Test successful speech recognition"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "тестовый текст"}
        mock_post.return_value = mock_response

        # Test parameters
        test_params = {
            "audio_file": BytesIO(self.create_test_wav()),
            "language": "ru-RU",
            "topic": "general",
        }

        # Invoke tool
        results = list(self.tool._invoke(test_params))

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(hasattr(result, "message"))
        self.assertEqual(result.message.text, "тестовый текст")

    @patch("requests.post")
    def test_no_speech_detected(self, mock_post):
        """Test handling of no speech detected"""
        # Mock API response with empty result
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": ""}
        mock_post.return_value = mock_response

        test_params = {
            "audio_file": BytesIO(self.create_test_wav()),
            "language": "ru-RU",
            "topic": "general",
        }

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.message.text, "No speech detected in the audio file")

    @patch("requests.post")
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = mock_response

        test_params = {
            "audio_file": BytesIO(self.create_test_wav()),
            "language": "ru-RU",
            "topic": "general",
        }

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("API error: 401", result.message.text)

    def test_missing_audio_file(self):
        """Test handling of missing audio file"""
        test_params = {"language": "ru-RU", "topic": "general"}

        results = list(self.tool._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.message.text, "Error: No audio file provided")

    def test_missing_api_key(self):
        """Test handling of missing API key"""
        tool_without_key = SpeechToTextTool.from_credentials({})
        test_params = {
            "audio_file": BytesIO(self.create_test_wav()),
            "language": "ru-RU",
            "topic": "general",
        }

        results = list(tool_without_key._invoke(test_params))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.message.text, "Error: API key is required")

    def test_language_parameters(self):
        """Test different language parameters"""
        test_languages = ["ru-RU", "en-US", "tr-TR", "uk-UA"]

        for lang in test_languages:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"result": "test result"}
                mock_post.return_value = mock_response

                test_params = {
                    "audio_file": BytesIO(self.create_test_wav()),
                    "language": lang,
                    "topic": "general",
                }

                list(self.tool._invoke(test_params))

                # Verify API was called with correct language
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["params"]["lang"], lang)

    def test_topic_parameters(self):
        """Test different topic parameters"""
        test_topics = ["general", "maps", "dates", "names", "numbers"]

        for topic in test_topics:
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"result": "test result"}
                mock_post.return_value = mock_response

                test_params = {
                    "audio_file": BytesIO(self.create_test_wav()),
                    "language": "ru-RU",
                    "topic": topic,
                }

                list(self.tool._invoke(test_params))

                # Verify API was called with correct topic
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[1]["params"]["topic"], topic)


if __name__ == "__main__":
    unittest.main()
