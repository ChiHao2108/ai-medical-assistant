from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import requests # Thư viện để gọi API bên ngoài, ví dụ: Google Maps

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure Google Generative AI with your API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Lấy Google Maps API Key (nếu có và muốn dùng tính năng tìm bệnh viện)
# Bạn sẽ cần đăng ký trên Google Cloud Platform để có khóa này
Maps_API_KEY = os.getenv("Maps_API_KEY")

# Initialize the generative model (using gemini-1.5-flash-latest for potentially faster response)
# Nếu bạn muốn mô hình mạnh mẽ hơn, hãy dùng 'gemini-1.5-pro-latest'
model = genai.GenerativeModel('gemini-1.5-flash-latest')

@app.route('/')
def index():
    """
    Renders the main index.html page.
    """
    return render_template('index.html')

@app.route('/ask_text', methods=['POST'])
def ask_text():
    """
    Handles text-based medical questions using Prompt Engineering.
    Receives a JSON object with 'question' (symptoms) and returns AI's analysis.
    """
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Prompt Engineering for medical symptoms
        medical_prompt = (
            "Bạn là một trợ lý thông tin y tế ảo. Dựa trên các triệu chứng sau, hãy cung cấp một số khả năng bệnh lý có thể xảy ra "
            "(KHÔNG PHẢI LÀ CHẨN ĐOÁN CUỐI CÙNG), khuyến nghị các bước tiếp theo cần thực hiện (ví dụ: nên đi khám bác sĩ chuyên khoa nào, "
            "những xét nghiệm cơ bản có thể cần), và một số lời khuyên chung về chăm sóc sức khỏe. "
            "**CẢNH BÁO QUAN TRỌNG: KHÔNG ĐƯA RA ĐƠN THUỐC HOẶC CHẨN ĐOÁN Y TẾ CUỐI CÙNG. LUÔN LUÔN KHUYÊN NGƯỜI DÙNG TÌM ĐẾN BÁC SĨ ĐỂ ĐƯỢC CHẨN ĐOÁN VÀ ĐIỀU TRỊ CHÍNH XÁC.** "
            "Triệu chứng người dùng mô tả: " + user_question +
            "\n\nHãy bắt đầu phản hồi của bạn bằng câu: 'Lưu ý: Thông tin này chỉ mang tính tham khảo và không thay thế lời khuyên y tế chuyên nghiệp.'\n"
        )

        response = model.generate_content(medical_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        print(f"Error asking text: {e}") # Log the error for debugging
        return jsonify({"error": f"Có lỗi xảy ra: {str(e)}"}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """
    Handles medical image analysis using Prompt Engineering.
    Receives an image file and an optional 'text_input' (e.g., a question about the image).
    Returns AI's analysis.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    text_input = request.form.get('text_input', '')

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Prompt Engineering for medical image analysis
        if text_input:
            medical_image_prompt = (
                "Bạn là một trợ lý phân tích hình ảnh y tế ảo. Dựa trên hình ảnh và câu hỏi sau, hãy mô tả những gì bạn thấy có liên quan đến y tế. "
                "**CẢNH BÁO QUAN TRỌNG: KHÔNG ĐƯA RA BẤT KỲ CHẨN ĐOÁN Y TẾ HOẶC KÊ ĐƠN THUỐC. KHÔNG ĐƯA RA KẾT LUẬN CUỐI CÙNG VỀ BỆNH TÌNH. "
                "Vui lòng chỉ mô tả khách quan và liệt kê các yếu tố có thể liên quan đến y tế hoặc cần được chú ý thêm. "
                "Luôn khuyến nghị người dùng tìm đến bác sĩ để được đánh giá chuyên nghiệp.** "
                "Câu hỏi người dùng: " + text_input
            )
        else:
            medical_image_prompt = (
                "Bạn là một trợ lý phân tích hình ảnh y tế ảo. Vui lòng mô tả những gì bạn thấy trong hình ảnh này có liên quan đến y tế. "
                "**CẢNH BÁO QUAN TRỌNG: KHÔNG ĐƯA RA BẤT KỲ CHẨN ĐOÁN Y TẾ HOẶC KÊ ĐƠN THUỐC. KHÔNG ĐƯA RA KẾT LUẬN CUỐI CÙNG VỀ BỆNH TÌNH. "
                "Vui lòng chỉ mô tả khách quan và liệt kê các yếu tố có thể liên quan đến y tế hoặc cần được chú ý thêm. "
                "Luôn khuyến nghị người dùng tìm đến bác sĩ để được đánh giá chuyên nghiệp.** "
            )
        contents = [medical_image_prompt, image]

        response = model.generate_content(contents)
        return jsonify({"analysis": response.text})
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({"error": f"Có lỗi xảy ra trong quá trình phân tích hình ảnh: {str(e)}"}), 500

@app.route('/get_nearby_hospitals', methods=['POST'])
def get_nearby_hospitals():
    """
    Mô phỏng chức năng tìm bệnh viện gần nhất bằng Google Maps Places API.
    **Lưu ý:** Chức năng này yêu cầu Maps_API_KEY hợp lệ và có thể phát sinh phí.
    """
    if not Maps_API_KEY:
        return jsonify({"error": "Google Maps API Key not configured."}), 500

    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not latitude or not longitude:
        return jsonify({"error": "Location (latitude, longitude) not provided."}), 400

    # Google Maps Places API endpoint (Find Place from Text or Nearby Search)
    # Đây là ví dụ sử dụng Nearby Search. radius tính bằng mét.
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius=5000&type=hospital&key={Maps_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        places_data = response.json()

        hospitals = []
        if places_data and 'results' in places_data:
            for place in places_data['results']:
                hospitals.append({
                    "name": place.get('name'),
                    "address": place.get('vicinity'),
                    "rating": place.get('rating'),
                    "open_now": place.get('opening_hours', {}).get('open_now')
                })
        return jsonify({"hospitals": hospitals})
    except requests.exceptions.RequestException as e:
        print(f"Error calling Google Places API: {e}")
        return jsonify({"error": f"Could not retrieve hospital information: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error in get_nearby_hospitals: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Run the Flask application
    # Set debug=False for production environments for security and performance
    app.run(debug=True)