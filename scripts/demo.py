from huggingface_hub import hf_hub_download
import torch
import os
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys
sys.path.append('..')
from src.utils import FlamingoProcessor
from demo_utils import image_paths, clean_generation
from tqdm import tqdm
import json



def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f,indent=4)
    f.close()


def test_single_data_thumbnail(vqa_model, processor, img_dir):

    thumb_short_img = os.path.join(img_dir, 'thumb_short.png')
    assert os.path.exists(thumb_short_img), '[Error]: thumb_short doesnt exist!'

    prompt1 = """
        Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy, 
        and you are allowed to use any necessary information on the Internet for you to answer questions. 
        For now I am giving you a set of MRI scaning slices of neonatal brains, 
        these slices are marked with coressponding slice labels, like 'Slice 10' and 'Slice 11'. 
        The label means the slice depth of this slice, 
        for example, 'Slice 11' is in the middle layer between 'Slice 10' and 'Slice 12'. 
        They are presented in thumbnail format. 
        Follow the examples and answer the last question.
        Question: What is the severity level of brain injury in this ADC? Answer: level1.<|endofchunk|>
        Question: What is the severity level of brain injury in this ADC? Answer: level2.<|endofchunk|>
        Question: What is the severity level of brain injury in this ADC? Answer: level3.<|endofchunk|>
        Question: What is the severity level of brain injury in this ADC? Answer: level4.<|endofchunk|>
    """

    question1_prompt = """
        You need to answer questions in the order they are given, and output in the predefined rules.
        For [Lesion Existence] questions, you need to decide the existence of the leison, and answer with 'yes', Or 'no'.
        [Lesion Existence] Does a lesion exist in this brain? Specify the slice number.
        [/INST]
    """

    question2_prompt = """
        For this question, you need to judge the lesion level of the brain MRI slices, the rule is: 
            if the lesion region percentage <= 0.01, answer with 'level1',
            if 0.01< lesion region percentage <=0.05, answer with 'level2',
            if 0.05< lesion region percentage <=0.5, answer with 'level3',
            if 0.5< lesion region percentage <=1.0, answer with 'level4'.
        Select one answer from level1, level2, level3 or level4.
        <image>Question: What is the severity level of brain injury in this ADC? Answer:
    """
    question3_prompt = """
        You need to answer questions in the order they are given, and output in the predefined rules.
        For [Scanner Type] questions, you need to decide the given MRI slice is scanned by GE 1.5T or SIEMENS 3T, and answer with '1.5T' or '3T'
        [Scanner Type] What is the Scanner Type of this ADC? 
    """
    question_list = [
        # question1_prompt, 
        question2_prompt, 
        # question3_prompt
        ]
    answer_dict = {}
    for cur_idx, cur_ques in enumerate(question_list):
        img_list = []
        prompt_list = []

        raw_image = Image.open(thumb_short_img).convert("RGB")
        img_list.append(raw_image)
        # prompt_list.append(f'thumbnail: <image>')

        image_tensor = processor.preprocess_images(img_list)
        image_tensor = repeat(image_tensor, 'N c h w -> b N T c h w', b=1, T=1)

        prompt_list = [prompt1] + prompt_list + [cur_ques]
        # 'slice 7': <image>, 'slice 8' : <image>
        # prompt_list = [prompt1] + ['<image>'] + [cur_ques]
        prompt = ''.join(''.join(prompt_list).split('\n'))
        tokenized_data = processor.encode_text(prompt)

        for _ in range(20):
            try: 
                with torch.inference_mode():
                    output_ids = vqa_model.generate(
                        vision_x=image_tensor.cuda(),
                        lang_x=tokenized_data["input_ids"].cuda(),
                        attention_mask=tokenized_data["attention_mask"].cuda(),
                        max_new_tokens=10,
                    )
            
                response = processor.tokenizer.decode(output_ids[0])
                response = clean_generation(response)
                outputs = response

                # import pdb; pdb.set_trace()
                if outputs == '':
                    raise ValueError
            except ValueError as e:
                print("[Warning] Answer not accept! Regenerate again!")
                continue
        print('##################################')
        print(outputs)
        print('##################################')
        short_ans = outputs.split('Answer:')[-1].split('.')[0]
        print(f"[shorted ans] ans:{short_ans}")
        ans_name = f'ans{cur_idx}'
        # answer_dict[ans_name] = outputs
        answer_dict[ans_name] = short_ans
        # answer_dict['range'] = [start_frm, end_frm]
    # import pdb; pdb.set_trace()
    return answer_dict


def main():
    accelerator = Accelerator() #when using cpu: cpu=True

    device = accelerator.device
    
    print('Loading model..')

    # >>> add your local path to Llama-7B (v1) model here:
    # llama_path = '../models/llama-7b-hf'
    llama_path = '../decapoda-research-llama-7B-hf'
    # llama_path = '/nobackup/users/zfchen/code/HIEVQA/med-flamingo/decapoda-research-llama-7B-hf/tokenizer.model'
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    """
    Step 1: Load images
    """
    image_paths = ['/nobackup/users/zfchen/code/HIEVQA/med-flamingo/img/synpic57813.jpg']

    demo_images = [Image.open(path) for path in image_paths]

    """
    Step 2: Define multimodal few-shot prompt 
    """

    # example few-shot prompt:
    # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Where is the largest mass located in the cerebellum? Answer:"
    # prompt = "\
    #     You are a helpful medical assistant. \
    #     You are being provided with images, a question about the image and an answer. \
    #     Follow the examples and answer the last question. \
    #         <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|>\
    #         <image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|>\
    #         <image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|>\
    #         <image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|>\
    #         <image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|>\
    #         <image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|>\
    #         <image>Question: Where is the largest mass located in the cerebellum? Answer:"
    
    prompt = "You are a helpful medical assistant. \
        You are being provided with images, a question about the image and an answer. \
            <image>Question:What is in the image? Answer:"

    """
    Step 3: Preprocess data 
    """
    print('Preprocess data')
    pixels = processor.preprocess_images(demo_images)
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
    tokenized_data = processor.encode_text(prompt)

    """
    Step 4: Generate response 
    """

    # actually run few-shot prompt through model:
    print('Generate from multimodal few-shot prompt')
    generated_text = model.generate(
        vision_x=pixels.to(device),
        lang_x=tokenized_data["input_ids"].to(device),
        attention_mask=tokenized_data["attention_mask"].to(device),
        # max_new_tokens=10,
        max_new_tokens=512,
    )
    response = processor.tokenizer.decode(generated_text[0])
    response = clean_generation(response)

    print(f'{response=}')



def main_new():
    accelerator = Accelerator() #when using cpu: cpu=True

    device = accelerator.device
    
    print('Loading model..')

    # >>> add your local path to Llama-7B (v1) model here:
    # llama_path = '../models/llama-7b-hf'
    llama_path = '../decapoda-research-llama-7B-hf'
    # llama_path = '/nobackup/users/zfchen/code/HIEVQA/med-flamingo/decapoda-research-llama-7B-hf/tokenizer.model'
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    dataset_path = '/nobackup/users/zfchen/code/HIEVQA/HIE_eval/MedicalEval/VLP_web_data/HIE_VQA'
    answers_file = '/nobackup/users/zfchen/code/HIEVQA/med-flamingo/answers'
    mgh_dir = os.path.join(dataset_path, 'MGH')
    dataset_split = ['BONBID2023_Train', 'BONBID2023_Test']
    for data_split in dataset_split:
        cur_split_data_dir = os.path.join(mgh_dir, data_split, '1ADC_ss')
        img_dirs = sorted(os.listdir(cur_split_data_dir))
        cur_split_answer = {}
        cur_answer_file = os.path.join(
            answers_file, 
            f'{data_split}_v1.json'
            # f'{data_split}_v_thumbnail.json'
            # f'{data_split}_debug.json'
            )

        for img_dir_idx, img_dir in enumerate(tqdm(img_dirs)):
            if not img_dir.startswith('MGHNICU'):
                continue
            
            answer = test_single_data_thumbnail(
                vqa_model=model,
                processor=processor,
                img_dir=os.path.join(cur_split_data_dir, img_dir)
            )
            data_id = img_dir.split('/')[-1].split('-')[0]
            cur_split_answer[data_id] = answer
            if img_dir_idx % 2 == 0:
                jsondump(cur_answer_file, cur_split_answer)
            
            # del model
            # torch.cuda.empty_cache()
            
        jsondump(cur_answer_file, cur_split_answer)


if __name__ == "__main__":

    # main()
    main_new()
