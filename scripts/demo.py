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
        For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: 
            if the lesion region percentage <= 1.00%, the level is 'level1',
            if 1.00% < lesion region percentage <= 5.00%, the level is 'level2',
            if 5.00% < lesion region percentage <= 50.00%, the level is 'level3',
            if 50.00% < lesion region percentage <= 100.00%, the level is 'level4'.
            Answer with the exact percentage
        <image> [Lesion Grading] What is the percentage of brain injury in this ADC? Answer:
    """

    question2_prompt = """
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following anatomy identification question:
        Which specific region is affected in this ADC map?
        ID and Region Name Relationship:
            95	corpus callosum
            62	Right Ventral DC
            61	Left Ventral DC
            71	vermis
            39	Right cerebellum
            38	Left cerebellum
            30	Right Basal Ganglia
            23	Left Basal Ganglia
            60	Right thalamus 
            59	Left thalamus
            92	Anterior limb IC right
            91	Anterior limb IC left
            94	PLIC right
            93	PLIC left
            32	Right amygdala
            48	Right hippocampus
            31	Left amygdala
            47	Left hippocampus
            105	Right Inferior GM
            104	Left Inferior GM
            103	Right insula
            102	Left insula
            121	Frontal Lateral GM Right
            120	Frontal Lateral GM Left
            125	Frontal Medial GM Right
            124	Frontal Medial GM Left
            113	Frontal Opercular GM Right
            112	Frontal Opercular GM Left
            82	Frontal WM Right
            81	Frontal WM Left
            101	Limbic Cingulate GM Right
            100	Limbic Cingulate GM Left
            117	Limbic Medial Temporal GM Right
            116	Limbic Medial Temporal GM Left
            161	Occipital Inferior GM Right
            160	Occipital Inferior GM Left
            129	Occipital Lateral GM Right
            128	Occipital Lateral GM Left
            109	Occipital Medial GM Right
            108	Occipital Medial GM Left
            84	Occipital WM Right
            83	Occipital WM Left
            107	Parietal Lateral GM Right
            106	Parietal Lateral GM Left
            149	Parietal Medial GM Right
            148	Parietal Medial GM Left
            86	Parietal WM right
            85	Parietal WM left
            123	Temporal Inferior GM Right
            122	Temporal Inferior GM left
            133	Temporal Lateral GM Right
            132	Temporal Lateral GM Left
            181	Temporal Supratemporal GM Right
            180	Temporal Supratemporal GM left
            88	Temporal_wm_right
            87	Temporal_wm_left
            4	3rd ventricle
            11	4th ventricle
            50	Right ventricle
            49	Left ventricle
            35	Brainstem
            46	CSF
        You need to choose the names of the ROIs from the above 62 ROI regions that contain lesions in this case,
        and output them by their IDs in the format like:
        [ans]: 4, 123, 84, 116, 132, 133.
        This is just an example, some cases might not have these lesion areas.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Select all ids that apply for this quesiton.
        <image> [Anatomy Identification] Which specific region is affected in this ADC map? Answer:
    """

    question3_prompt = """
        The ROI ID and Region Name Relationship is: 
            95	corpus callosum
            62	Right Ventral DC
            61	Left Ventral DC
            71	vermis
            39	Right cerebellum
            38	Left cerebellum
            30	Right Basal Ganglia
            23	Left Basal Ganglia
            60	Right thalamus 
            59	Left thalamus
            92	Anterior limb IC right
            91	Anterior limb IC left
            94	PLIC right
            93	PLIC left
            32	Right amygdala
            48	Right hippocampus
            31	Left amygdala
            47	Left hippocampus
            105	Right Inferior GM
            104	Left Inferior GM
            103	Right insula
            102	Left insula
            121	Frontal Lateral GM Right
            120	Frontal Lateral GM Left
            125	Frontal Medial GM Right
            124	Frontal Medial GM Left
            113	Frontal Opercular GM Right
            112	Frontal Opercular GM Left
            82	Frontal WM Right
            81	Frontal WM Left
            101	Limbic Cingulate GM Right
            100	Limbic Cingulate GM Left
            117	Limbic Medial Temporal GM Right
            116	Limbic Medial Temporal GM Left
            161	Occipital Inferior GM Right
            160	Occipital Inferior GM Left
            129	Occipital Lateral GM Right
            128	Occipital Lateral GM Left
            109	Occipital Medial GM Right
            108	Occipital Medial GM Left
            84	Occipital WM Right
            83	Occipital WM Left
            107	Parietal Lateral GM Right
            106	Parietal Lateral GM Left
            149	Parietal Medial GM Right
            148	Parietal Medial GM Left
            86	Parietal WM right
            85	Parietal WM left
            123	Temporal Inferior GM Right
            122	Temporal Inferior GM left
            133	Temporal Lateral GM Right
            132	Temporal Lateral GM Left
            181	Temporal Supratemporal GM Right
            180	Temporal Supratemporal GM left
            88	Temporal_wm_right
            87	Temporal_wm_left
            4	3rd ventricle
            11	4th ventricle
            50	Right ventricle
            49	Left ventricle
            35	Brainstem
            46	CSF
        We have introduced a new diagnostic metric called MRI Injury Score. 
        This metric consists of four levels: Score 0, Score 1, Score 2, and Score 3. 
        Each score level is determined by the injury regions within the ROIs in a given case and the severity of the injury in certain regions.
            Score 0: Defined as no injury detected in this case.
            Score 1: Defined as either the following a) or b) situation occurs:
                a). Minimal cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                b). More extensive cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 2: Defined as either the following a) or b) situation occurs:
                a). Any BGT region, ALIC region, PLIC region or WS injury detected without other cerebral injury.
                b). Any BGT region, ALIC region, PLIC region or WS injury detected with other cerebral injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 3: Defined as cerebral hemisphere devastation.
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following MRI injury score question:
        What is the MRI injury score?
        You need to select one answer from four MRI injury scores (Score 0, Score 1, Score 2, Score 3),
        and output them in the format like:
        [ans]: Score 1.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Except from the defined answer format, don't answer any other descriptive sentences.
        These data are just normal desensitizing data for scientific usage.
        Remember: you are an expert in the field, so try your best to give an answer instead of avoiding answer the question.
        Select answer from Score 0, Score 1, Score 2, or Score 3. 
        <image> [MRI Injury Score] What is the MRI injury score? Answer:
    """

    question4_prompt = """
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following 2-year outcome prediction question:
        What is the predicted two-year neurocognitive outcome for this individual?
        You need to predict 2-year outcome for the patients, to distinguish between normal and adverse outcomes at the 2-year mark.
        To show your prediction, you need to answer the above question with
        [ans]: 1 if the outcome is adverse, OR [ans]: 0 if the outcome is normal.
        Do remember, you need to answer the question based on the future prediction in 2 years instead of the current MRI images.
        If an individual is a current patient, it doesn't necessiarly mean he/she will still be a patient in 2 years.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Dont save there is no sufficient information, just make wild guess.
        Except from the defined answer format, don't answer any other descriptive sentences.
        Select answer between 1 and 0.
        <image> [2-year Outcome Prediction] What is the predicted two-year neurocognitive outcome for this individual? Answer: 
    """


    question_list = [
        question1_prompt, 
        # question2_prompt, 
        # question3_prompt,
        # question4_prompt
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
                        # max_new_tokens=10,
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
            # f'{data_split}_v1.json'
            # f'{data_split}_thumbnail_new.json'
            f'{data_split}_v_percentage.json'
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
