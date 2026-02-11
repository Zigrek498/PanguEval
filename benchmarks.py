from utils import (
    MMLU,
    CMMLU,
    MMLU_Pro,
    CEval,
    GPQA_Diamond,
    MATH_500,
    AIME24,
    AIME25,
    CSarcasm,
    Wbdmt,
    Dwsxjy,
    Wbsxcq,
    Yxxxtq_inf
    )

def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):
    supported_dataset = ["MMLU","CMMLU","MMLU_Pro","CEval","GPQA_Diamond","MATH_500","AIME24","AIME25","CSarcasm","Wbdmt","Dwsxjy","Wbsxcq","Yxxxtq_inf"]

    if eval_dataset == "MMLU":
        dataset = MMLU(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CMMLU":
        dataset = CMMLU(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "MMLU_Pro":
        dataset = MMLU_Pro(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CEval":
        dataset = CEval(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "GPQA_Diamond":
        dataset = GPQA_Diamond(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MATH_500":
        dataset = MATH_500(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "AIME24":
        dataset = AIME24(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "AIME25":
        dataset = AIME25(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CSarcasm":
        dataset = CSarcasm(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "Wbdmt":
        dataset = Wbdmt(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Dwsxjy":
        dataset = Dwsxjy(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Wbsxcq":
        dataset = Wbsxcq(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Yxxxtq_inf":
        dataset = Yxxxtq_inf(model,eval_dataset_path,eval_output_path)
    
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None

    return dataset

if __name__ == '__main__':
    pass    