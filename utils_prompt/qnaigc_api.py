#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import traceback
import time
import os
import random
from openai import OpenAI
from dotenv import find_dotenv,load_dotenv
_=load_dotenv(find_dotenv())

class QNAIGCDeepSeek:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_API_BASE')
        self.model = os.getenv("OPENAI_Model")
        print(self.base_url)
        print(self.api_key)
        if not self.api_key:
            raise ValueError(
                "API key is required. Please provide api_key parameter or set QNAIGC_API_KEY environment variable")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )


    def chat_completion(self, messages, max_tokens: int = 4096,temperature: float = 0.3):
        for attempt in range(1, 6):          # 最多 5 次
           try:
              response = self.client.chat.completions.create(
                  model=self.model,
                  messages=messages,
                  temperature=temperature
            )
              return response.choices[0].message.content
           except Exception as e:
               wait = 2 ** attempt + random.uniform(0, 1)
               print(f"警告  第{attempt}次失败: {e}，{wait:.1f}s 后重试")
               time.sleep(wait)
 
        print("错误 重试用完，返回空串")
        return ""
        # try:
        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=messages,
        #         max_tokens=max_tokens,   # 官方支持的参数
        #         temperature=temperature
        #     )
        #     return response.choices[0].message.content
        # except Exception as e:
        #     print(f"模型请求错误{traceback.format_exc()}")
        #     return ""

    def generate_sentence_from_prompt(self, keypoint_prompt: str,max_tokens: int = 4096, temperature: float = 0.6):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that generates review sentences based on given prompts and examples. "
                    "Strictly use the exact aspect terms provided in the prompt; do NOT paraphrase or alter them. "
                    "Ensure the sentence clearly expresses the sentiment for each aspect as instructed."
                )
            },
            {
                "role": "user",
                "content": f"Based on the following prompt and examples, please generate a sentence and its corresponding label. "
                           f"Strictly use the aspect terms provided. Return in the format:\n"
                           f"[Generated sentence]\nLabel: [aspect-label pairs]\n\n{keypoint_prompt}\n\nGenerated output:"
            }
        ]

        try:
            result = self.chat_completion(messages[:], max_tokens=max_tokens, temperature=temperature)

            if not result:
                return {"success": False, "sentence": "", "error": "Empty response from model"}

            # 解析大模型返回的结果
            lines = result.strip().split('\n')
            sentence = ""
            label = ""

            # 查找句子和Label部分
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("Label:"):
                    # 前面的内容是句子
                    sentence = '\n'.join(lines[:i]).strip()
                    # Label及之后的内容
                    label = '\n'.join(lines[i:]).strip()
                    break
            else:
                # 如果没有找到"Label:"，假设整个结果都是句子
                sentence = result.strip()
                label = "Label: []"

            return {
                "success": True,
                "sentence": sentence,
                "label": label,
                "full_output": result
            }

        except Exception as e:
            return {
                "success": False,
                "sentence": "",
                "error": str(e)
            }

    def batch_generate(self, prompts: list, max_tokens: int = 150, temperature: float = 0.3,
                       delay: float = 1.0) -> list:
        results = []

        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i + 1}/{len(prompts)}...")

            result = self.generate_sentence_from_prompt(
                keypoint_prompt=prompt
            )

            results.append({
                "prompt_index": i,
                "prompt": prompt,
                "result": result
            })

            if i < len(prompts) - 1:
                time.sleep(delay)

        return results


def create_qnaigc_client() -> QNAIGCDeepSeek:
    return QNAIGCDeepSeek()
