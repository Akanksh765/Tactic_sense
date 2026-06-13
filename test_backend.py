import os
import traceback

# Enforce the same stability rules as the main app
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

print("[DEBUG] Booting backend test...")

try:
    import torch
    from transformers import XLMRobertaTokenizer
    from src.model import TacticSenseModel
    from src.predict import predict_tactic
    
    print("[DEBUG] Libraries loaded successfully.")

    # 1. Load Resources
    print("[DEBUG] Loading Tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    print("[DEBUG] Loading Model Weights...")
    model = TacticSenseModel()
    # Apply your custom strict=False loading logic here if needed
    model.load_state_dict(torch.load("models/tacticsense_v1.pt", map_location='cpu'), strict=False)
    model.eval()
    print("[DEBUG] Model ready.")

    # 2. Simulate the 5-Turn Conversation
    turns = [
        "Hi, I am very interested in your mountain bike.",
        "Great! Yes, it is available and in mint condition.",
        "Awesome. I am a student looking for something affordable. Can you do a discount?",
        "Sorry, my asking price is $300. I am pretty firm on that.",
        "If there is no flexibility, then I have to walk away. That is way past my budget."
    ]
    ids = [0, 1, 0, 1, 0] # 0 = Buyer, 1 = Seller

    print("[DEBUG] Running inference pass...")
    labels, probabilities = predict_tactic(turns, ids, model, tokenizer)
    
    print("\n[OK] SUCCESS! Backend is 100% stable.")
    print("Top Prediction:", labels[0], "-", probabilities[0])

except Exception as e:
    print("\n[FAIL] CRASH DETECTED IN BACKEND:")
    traceback.print_exc()