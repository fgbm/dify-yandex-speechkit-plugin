from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.text_to_speech import TextToSpeechTool


class YandexSpeechkitProvider(ToolProvider):
    """
    Yandex SpeechKit tool provider for speech synthesis and recognition
    """

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Validate provider credentials by testing the TTS API
        """
        api_key = credentials.get("api_key")

        if not api_key:
            raise ToolProviderCredentialValidationError("API key is required")

        if not isinstance(api_key, str) or len(api_key.strip()) == 0:
            raise ToolProviderCredentialValidationError("API key is required")

        # Test credentials with a simple TTS request
        try:
            # Create a TTS tool instance and test with minimal parameters
            tts_tool = TextToSpeechTool.from_credentials(credentials)

            # Test with a simple text synthesis
            test_params = {
                "text": "test",
                "voice": "marina",
                "emotion": "neutral",
                "speed": 1.0,
                "format": "mp3",
            }

            # Try to invoke the tool - this will validate the API key
            result_generator = tts_tool.invoke(tool_parameters=test_params)

            # Process the generator to actually make the API call
            results = list(result_generator)

            # Check if we got a successful result (should have text and audio blob)
            if not results:
                raise ToolProviderCredentialValidationError(
                    "No response from Yandex SpeechKit API"
                )

            # Look for error messages in the results
            for result in results:
                if hasattr(result, "message") and result.message:
                    message_text = getattr(result.message, "text", "")
                    if message_text.startswith("Error:"):
                        raise ToolProviderCredentialValidationError(
                            f"API validation failed: {message_text}"
                        )

        except ToolProviderCredentialValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to validation errors
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise ToolProviderCredentialValidationError(
                    "Invalid API key - unauthorized access"
                )
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                raise ToolProviderCredentialValidationError(
                    "API key does not have required permissions"
                )
            elif "timeout" in error_msg.lower():
                raise ToolProviderCredentialValidationError(
                    "API request timeout - please check your connection"
                )
            else:
                raise ToolProviderCredentialValidationError(
                    f"Credential validation failed: {error_msg}"
                )
