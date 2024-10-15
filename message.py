import random
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import FlexSendMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction, CarouselContainer, BubbleContainer, BoxComponent, TextComponent, ButtonComponent, URIAction
from neo4j import GraphDatabase
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
import json
import ollama
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Initialize app and LINE API
app = Flask(__name__)
line_bot_api = LineBotApi('xTivPDr65I/9BmuQNkMg3yh2Y/yOf0VoZegoRq73O/5H8RpAg5YmKOsMoCdcFiOcrmwUFfBpF8NMYhw6piXwrHYyU8b7NvsBDR9g8byv+47BvSu0om4wGQh788vSzajYslQt7Xd2PL1SjZ+F87bNKgdB04t89/1O/w1cDnyilFU=')

# Initialize Neo4j connection
neo4j_uri = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "0986576621"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# Function to run Neo4j query
def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result]

# Function to save conversation history in Neo4j
def save_conversation_to_neo4j(user_id, message, response):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            CREATE (m:Message {text: $message, timestamp: timestamp()})
            CREATE (r:Response {text: $response, timestamp: timestamp()})
            MERGE (u)-[:SENT]->(m)
            MERGE (u)-[:RECEIVED]->(r)
            """, 
            user_id=user_id, message=message, response=response
        )

# Function to encode a query and check similarity
def compute_similar(query, corpus):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)
    max_score_idx = np.argmax(cosine_scores.cpu().numpy())
    return corpus[max_score_idx], cosine_scores[0, max_score_idx].item()

# Function to handle user input using similarity
def handle_greeting_or_category(msg):
    # Query for greetings and categories from Neo4j
    greetings_query = "MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply"
    greetings = run_query(greetings_query)
    greeting_corpus = [record['name'] for record in greetings]
    
    best_match, score = compute_similar(msg, greeting_corpus)
    
    # Check if similarity score is above a threshold (set 0.5 for now)
    if score >= 0.95:
        reply = next(record['reply'] for record in greetings if record['name'] == best_match)
        return reply
    else:
        return "ขอโทษครับ ไม่พบข้อมูลที่คล้ายคลึงกับคำที่คุณพิมพ์"

# Function to ensure valid URL encoding
def ensure_http(url, base_url):
    full_url = urljoin(base_url, url)
    return quote(full_url, safe=':/')

# Function to scrape product details from Nike website with sorting
def scrape_nike_sorted(url, sort_order='ascending'):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    products_details = []
    products = soup.find_all("figure")

    for product_element in products:
        product_name = product_element.find("div", class_="product-card__title").text.strip()
        product_normal_price = product_element.find("div", {"data-testid": "product-price"}).text.strip().replace(",", "").replace("฿", "")
        product_price = float(product_normal_price) if product_normal_price else 0
        product_link_tag = product_element.find("a", class_="product-card__link-overlay")
        relative_url = product_link_tag.get("href") if product_link_tag else None
        product_url = ensure_http(relative_url, url)

        products_details.append({
            'product_name': product_name,
            'normal_price': product_price,
            'product_url': product_url
        })

    # Sort the products based on the sort_order
    if sort_order == 'ascending':
        products_details.sort(key=lambda x: x['normal_price'])
    else:
        products_details.sort(key=lambda x: x['normal_price'], reverse=True)

    return products_details

# Function to send Flex message with sorted product list
def send_flex_message(reply_token, products):
    if not products:
        text_message = TextSendMessage(text="No products found.")
        line_bot_api.reply_message(reply_token, text_message)
        return

    products = products[:10]  # Limit to 10 products
    bubbles = [
        BubbleContainer(
            body=BoxComponent(
                layout="vertical",
                contents=[
                    TextComponent(text=prod['product_name'], weight="bold", size="md", wrap=True),
                    TextComponent(text=f"ราคาปกติ: {prod['normal_price']:.2f} บาท", size="sm", color="#999999")
                ]
            ),
            footer=BoxComponent(
                layout="vertical",
                contents=[
                    ButtonComponent(
                        style="primary",
                        height="sm",
                        action=URIAction(label="กดเพื่อดูรูป", uri=prod['product_url'])
                    )
                ]
            )
        ) for prod in products
    ]
    carousel = CarouselContainer(contents=bubbles)
    
    flex_message = FlexSendMessage(alt_text="Product List", contents=carousel)

    # ส่ง Flex Message พร้อม Quick Reply สำหรับ Discount, New Arrivals, Best Seller
    quick_reply = QuickReply(items=[
        QuickReplyButton(action=URIAction(label="Discount", uri="https://www.nike.com/th/w/sale-3yaep")),
        QuickReplyButton(action=URIAction(label="New Arrivals", uri="https://www.nike.com/th/w/new-shoes-3n82yzy7ok")),
        QuickReplyButton(action=URIAction(label="Best seller", uri="https://www.nike.com/th/w/best-76m50")),
        QuickReplyButton(action=MessageAction(label="เลือกรองเท้าใหม่อีกครั้ง", text="เลือกรองเท้าใหม่อีกครั้ง"))  # Shortened label
    ])

    line_bot_api.reply_message(
        reply_token,
        messages=[flex_message, TextSendMessage(text="เลือกหัวข้อที่สนใจเพิ่มเติม:", quick_reply=quick_reply)]
    )

# Function to send Quick Reply for sorting options
def send_sorting_options(reply_token):
    quick_reply = QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="เรียงจากน้อยไปมาก", text="เรียงจากน้อยไปมาก")),
        QuickReplyButton(action=MessageAction(label="เรียงจากมากไปน้อย", text="เรียงจากมากไปน้อย")),
    ])

    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text="กรุณาเลือกการจัดเรทราคา : ", quick_reply=quick_reply)
    )

# Function to retrieve greetings from Neo4j
def get_greetings():
    cypher_query = '''
    MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
    '''
    greeting_corpus = []
    with driver.session() as session:
        results = session.run(cypher_query)
        for record in results:
            greeting_corpus.append({
                'name': record['name'],
                'reply': record['reply']
            })
    return greeting_corpus

# Main route to handle requests from LINE
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        reply_token = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId']
        msg = json_data['events'][0]['message']['text'].lower()

        # Initialize chat history for Ollama response generation
        history_chat = []

        # ตรวจสอบว่าผู้ใช้กดปุ่ม "เลือกรองเท้าใหม่อีกครั้ง"
        if msg == "เลือกรองเท้าใหม่อีกครั้ง":
            # แสดง Quick Reply สำหรับหมวดหมู่รองเท้าใหม่
            quick_reply = QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="สำหรับผู้ชาย", text="รองเท้าสำหรับผู้ชาย")),
                QuickReplyButton(action=MessageAction(label="สำหรับผู้หญิง", text="รองเท้าสำหรับผู้หญิง")),
                QuickReplyButton(action=MessageAction(label="สำหรับเด็กผู้ชาย", text="รองเท้าสำหรับเด็กผู้ชาย")),
                QuickReplyButton(action=MessageAction(label="สำหรับเด็กผู้หญิง", text="รองเท้าสำหรับเด็กผู้หญิง")),
            ])

            # ส่งข้อความพร้อม Quick Reply ใหม่
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="กรุณาเลือกหมวดหมู่รองเท้า:", quick_reply=quick_reply)
            )
            return 'OK'

        # Check if the user wants to sort the product list
        if msg in ["เรียงจากน้อยไปมาก", "เรียงจากมากไปน้อย"]:
            # กำหนด URL ของการเรียงลำดับ
            if msg == "เรียงจากน้อยไปมาก":
                url = "https://www.nike.com/th/w/mens-shoes-nik1zy7ok?sortBy=priceAsc"
            else:  # "เรียงจากมากไปน้อย"
                url = "https://www.nike.com/th/w/mens-shoes-nik1zy7ok?sortBy=priceDesc"
    
            # ทำการ scrape ข้อมูลจากหน้าเว็บตาม URL ที่กำหนด
            products = scrape_nike_sorted(url)

            # ส่ง Flex message พร้อมรายการสินค้าที่จัดเรียงแล้ว
            send_flex_message(reply_token, products)
            return 'OK'

        # Get greetings from Neo4j
        greetings = get_greetings()
        
        # Check if the message matches any greeting name in Neo4j
        for greeting in greetings:
            if msg in greeting['name'].lower():
                response = greeting['reply']
                
                # เพิ่ม quick reply สำหรับหมวดหมู่รองเท้า
                quick_reply = QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="สำหรับผู้ชาย", text="รองเท้าสำหรับผู้ชาย")),
                    QuickReplyButton(action=MessageAction(label="สำหรับผู้หญิง", text="รองเท้าสำหรับผู้หญิง")),
                    QuickReplyButton(action=MessageAction(label="สำหรับเด็กผู้ชาย", text="รองเท้าสำหรับเด็กผู้ชาย")),
                    QuickReplyButton(action=MessageAction(label="สำหรับเด็กผู้หญิง", text="รองเท้าสำหรับเด็กผู้หญิง")),
                ])
                
                # ส่งข้อความทักทายพร้อม QuickReply
                line_bot_api.reply_message(
                    reply_token, 
                    [TextSendMessage(text=response, quick_reply=quick_reply)]
                )
                
                # Save conversation history to Neo4j
                save_conversation_to_neo4j(user_id, msg, response)
                return 'OK'
        
        # Mapping URLs to user input
        url_map = {
            "รองเท้าสำหรับผู้ชาย": "https://www.nike.com/th/w/mens-shoes-nik1zy7ok",
            "รองเท้าสำหรับผู้หญิง": "https://www.nike.com/th/w/womens-shoes-5e1x6zy7ok",
            "รองเท้าสำหรับเด็กผู้ชาย": "https://www.nike.com/th/w/boys-shoes-4413nzy7ok",
            "รองเท้าสำหรับเด็กผู้หญิง": "https://www.nike.com/th/w/girls-shoes-6bnmbzy7ok",
        }

        # Scrape products based on user input
        products = scrape_nike_sorted(url_map.get(msg, ""))  
        
        # Generate chatbot response using Sentence Transformer for handling similar words
        chatbot_response = handle_greeting_or_category(msg)
        history_chat.append(f"User: {msg}")  # Save user input to chat history
        history_chat.append(f"Bot: {chatbot_response}")  # Save bot response to chat history
        
        # ส่งปุ่ม Quick Reply สำหรับการเลือกวิธีการเรียงลำดับก่อนแสดงผลลัพธ์
        send_sorting_options(reply_token)

        # Save conversation history to Neo4j
        response = f"เลือกเนื้อหาถัดไป : {chatbot_response}"
        save_conversation_to_neo4j(user_id, msg, response)

    except Exception as e:
        print(f"Error processing the LINE event: {e}")

    return 'OK'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
