from groq import Groq

client = Groq(api_key="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")


SABIT_PROMPT = """
Aşağıdaki haberin doğruluğunu değerlendir:

Haber: "{haber}"

Cevabın şu formatta olsun:
- Doğru mu?: (Evet / Hayır / Şüpheli)
- Kısa açıklama:
"""


def analiz_et(haber):
    prompt = SABIT_PROMPT.format(haber=haber)

    response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    while True:
        haber = input("\nHaber metnini girin ('q' ile çık): ")
        if haber.lower() == "q":
            break
        cevap = analiz_et(haber)
        print(" Yapay Zeka Cevabı:", cevap)
