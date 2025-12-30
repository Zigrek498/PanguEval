from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("models.openPangu_7b_vllm")
logger.setLevel(logging.INFO)

class OpenPangu:
    def __init__(self, model_path, args):
        logger.info(f"正在加载盘古7B模型：{model_path}")
        # 初始化vLLM引擎
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=16384,
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            tokenizer_mode="slow",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

        # 加载tokenizer（使用transformers的版本）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 设置生成参数
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        # 思考模式："think"(默认)/"no_think"/"auto_think"
        self.thinking_mode = args.thinking_mode
        
        # 创建vLLM的采样参数
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=False  # 保留特殊token用于分割思考内容
        )

    def process_messages(self, messages):
        """处理消息并转换为模型输入格式（与HF版本一致）"""
        new_messages = []
        if isinstance(messages, list) and "role" in messages[0]:
            new_messages = messages
        elif "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]}) 
        else:
            new_messages.append({"role": "user", "content": messages["prompt"]})

        if self.thinking_mode == "no_think":
            new_messages[-1]["content"] += " /no_think"
        elif self.thinking_mode == "auto_think":
            new_messages[-1]["content"] += " /auto_think"
        
        return new_messages

    def generate_output(self, messages):
        """生成单个输出（思考内容和回答内容）"""
        prompt = self.process_messages(messages)
        outputs = self.llm.chat([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return self._parse_generated_text(generated_text)

    def generate_outputs(self, messages_list):
        """批量生成输出（更高效）"""
        prompts = [self.process_messages(messages) for messages in messages_list]
        outputs = self.llm.chat(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(self._parse_generated_text(generated_text))
        
        return results

    def _parse_generated_text(self, text):
        """解析生成的文本，分割思考内容和回答内容"""
        thinking_content = text.split("[unused17]")[0].split("[unused16]")[-1].strip()
        content = text.split("[unused17]")[-1].split("[unused10]")[0].strip()
        # return thinking_content, content
        return content

if __name__ == "__main__":
    # 测试模型是否能正常加载和生成
    class Args:
        temperature = 0.0
        top_p = 1.0
        repetition_penalty = 1.0
        max_new_tokens = 64
        thinking_mode = "auto_think"

    model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1"
    model = OpenPangu(model_path, Args())

    prompt = [{"role": "user", "content": "你是谁？ /no_think"}]
    
    # outputs = model.llm.chat([prompt], model.sampling_params)
    # generated_text = outputs[0].outputs[0].text
    generated_text = model.generate_output(prompt)
    
    print("===== MODEL OUTPUT =====")
    # print(outputs)
    print(generated_text)
