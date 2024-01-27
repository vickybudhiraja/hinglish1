## 
## Hinglish Supported, Meta’s Llama-2-7b (fine-tuned using axolotl)
## (Nathan Raw) https://replicate.com/nateraw
##

import torch
from transformers import AutoModelForCausalLM, pipeline

if __name__ == "__main__":

    PROMPT_TEMPLATE = (
        f"Translate from english to hinglish:\n{{en}}\n---\nTranslation:\n"
    )
    model_id = "nousresearch/llama-2-7b-hf"
    peft_model_id = "nateraw/llama-2-7b-english-to-hinglish"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.load_adapter(peft_model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=model_id,
    )
    print('\n')
    print('\n')

    out = pipe(
        ## No one is more intelligent than other. Only you are more intelligent than anyone else.
        PROMPT_TEMPLATE.format(en="Let us eat out"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ## No one is more intelligent than other. Only you are more intelligent than anyone else.
        PROMPT_TEMPLATE.format(en="meet me outside"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')
    
    out = pipe(
        ##  Only you are more intelligent than anyone else.
        PROMPT_TEMPLATE.format(en="I am hungry"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ##  
        PROMPT_TEMPLATE.format(en="The food is very tasty"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ##  
        PROMPT_TEMPLATE.format(en="when is the match starting?"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ##  
        PROMPT_TEMPLATE.format(en="good resturants near me"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ##  
        PROMPT_TEMPLATE.format(en="are you using this?"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')

    out = pipe(
        ##  
        PROMPT_TEMPLATE.format(en="will you do your workouts today?"),
        return_full_text=True,
        do_sample=True,
        max_new_tokens=256
    )[0]['generated_text']
    print(out)
    print('\n')
