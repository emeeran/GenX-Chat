Certainly! I'll provide instructions for each major function of the app:

1. Starting a Chat:
   - Enter your message in the chat input at the bottom of the main screen.
   - Press Enter or click the send button to get a response from the AI.

2. Choosing an AI Provider:
   - In the sidebar, use the "Select Provider" dropdown to choose between Groq and OpenAI.

3. Selecting an AI Model:
   - Expand the "Model" section in the sidebar.
   - Use the "Choose Model" dropdown to select a specific AI model.

4. Adjusting Model Parameters:
   - In the "Model" section, use the sliders to adjust:
     - Max Tokens
     - Temperature
     - Top-p
     - Top-k
     - Frequency Penalty
     - Presence Penalty

5. Selecting a Persona:
   - Expand the "Persona" section in the sidebar.
   - Choose a predefined persona from the dropdown or select "Custom".
   - For custom persona, enter your desired persona description in the text area.

6. Enabling Audio Responses:
   - Expand the "Audio & Language" section in the sidebar.
   - Check the "Enable Audio Response" box.
   - Select your preferred language and voice/language code.

7. Uploading a File:
   - Expand the "File Upload" section in the sidebar.
   - Click "Browse files" or drag and drop a supported file (PDF, DOCX, TXT, MD, JPG, JPEG, PNG).

8. Enabling Summarization:
   - Expand the "Summarize" section in the sidebar.
   - Check the "Enable Summarization" box.
   - Choose your preferred summarization type from the dropdown.

9. Enabling Content Creation Mode:
   - Expand the "Content Generation" section in the sidebar.
   - Check the "Enable Content Creation Mode" box.
   - Select the desired content type from the dropdown.

10. Saving a Chat:
    - Enter a name for your chat in the "Chat Name" input field in the sidebar.
    - Click the "Save" button to save the current chat.

11. Loading a Saved Chat:
    - Use the "Load Chat History" dropdown in the sidebar to select a previously saved chat.

12. Deleting a Saved Chat:
    - Select a chat from the "Load Chat History" dropdown.
    - Click the "Delete" button to remove the selected chat.

13. Starting a New Chat:
    - Click the "New" button in the sidebar to clear the current chat and start fresh.

14. Retrying the Last Response:
    - Click the "Retry" button in the sidebar to regenerate the last AI response.

15. Exporting Chat History:
    - Expand the "Export" section in the sidebar.
    - Choose your preferred export format (MD or PDF).
    - Click the "Export Chat" button to download the chat history.

16. Changing the Color Scheme:
    - Use the "Color Scheme" dropdown at the bottom of the sidebar to switch between Light and Dark modes.

17. Viewing Usage Statistics:
    - Check the bottom of the sidebar to see metrics for Total Words, Total Tokens, and Estimated Cost.

Remember that some features may depend on the selected AI provider or model. Always ensure you have the necessary API keys set up in your environment variables for the providers you want to use.


This app is  capable of the following functionalities:

1. Multi-provider AI Chat:
   - Supports both Groq and OpenAI as AI providers
   - Real-time streaming of AI responses

2. Chat History Management:
   - Saving and loading chat sessions
   - Deleting saved chats
   - Exporting chat history as Markdown or PDF

3. File Processing:
   - Upload and process various file types (PDF, DOCX, TXT, MD, JPG, JPEG, PNG)
   - Extract text from images using OCR

4. Text-to-Speech:
   - Convert AI responses to speech
   - Support for multiple languages (English, Tamil, Hindi)
   - Different voice options for OpenAI

5. Language Translation:
   - Translate responses to different languages

6. Content Generation:
   - Create various types of content based on user prompts (e.g., short stories, articles)

7. Text Summarization:
   - Summarize uploaded content or chat conversations
   - Multiple summarization types (Main Takeaways, Bulleted Points, Concise Summary, Executive Summary)

8. Persona Customization:
   - Select from predefined AI personas
   - Create and use custom personas

9. Model Parameter Adjustment:
   - Fine-tune AI model parameters (temperature, max tokens, top-p, top-k, etc.)

10. UI Customization:
    - Toggle between light and dark color schemes

11. Audio Response:
    - Enable/disable audio responses
    - Select different voices or language codes

12. Token and Cost Tracking:
    - Monitor total words, tokens used, and estimated cost

13. Rate Limiting:
    - Implement API call rate limiting to prevent overuse

14. Error Handling and Logging:
    - Comprehensive error catching and logging

15. Asynchronous Operations:
    - Use of asyncio for improved performance in API calls and database operations

16. Database Integration:
    - SQLite database for persistent storage of chat history

17. File Upload:
    - Support for uploading various file types to incorporate into the chat

18. Multiple Model Support:
    - Option to choose from different AI models within each provider

This app combines chat functionality with advanced AI features, making it a versatile tool for various text-based AI interactions, content creation, and analysis tasks.