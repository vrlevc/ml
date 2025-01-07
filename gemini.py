import google.generativeai as genai

genai.configure(api_key="AIzaSyCnFY1W4jchaKfsSMseUi2jNdthzGz5CtI")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)