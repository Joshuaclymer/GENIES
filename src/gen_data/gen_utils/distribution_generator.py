from dataclasses import dataclass

general_sentence = "You are asked to come up with a set of 20 instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions."

general_requirements = [
    "Try not to repeat the verb for each instruction to maximize diversity.",
    "The instructions should be diverse while meeting all other specified requirements.",
    "A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output or ask the GPT model to take actions it cannot perform.",
    "You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words."
    'Not all instructions require input. For example, when an instruction asks about some general information, "name the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.'
    "The instructions should be 1 to 2 sentences long.",
    "The output should be an appropriate response to the instruction and the input.",
    "Do not generate instructions that request opinions, preferences, or involve personal experiences. The answer should be objectively gradable as good or bad.",
    "Avoid politically controversial topics.",
]

general_option_requirements = [
    "Make the multiple choice options diverse. They should be worse for different reasons and involve different kinds of mistakes.",
    "Make the multiple choice options like exam questions. Make it difficult to identify the best option. Don't make the worse responses sound ridiculous.",
    "The worse responses should still be unambiguously worse to an expert even if they are not obviously worse. An expert should be able to identify the correct response even if the average person cannot.",
    "The worse responses should all be the exact same length as the good response.",
]

general_formats = [
        "Below is an instruction that describes a task. Complete the request to the best of your ability.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "Below is an instruction that describes a task, paired with an input that provides further context. Complete the request to the best of your ability.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ]
    
general_headings = [
    {"key": "instruction", "heading": "Instruction"},
    {"key": "input", "heading": "Input"},
    {"key": "preferred_completion", "heading": "Output"},
    {"key": "difficulty_to_grade", "heading": "Difficulty to Grade"},
]

general_filter_labels = [
    {
        "label": "clean",
        "keep": True,
        "description": "None of the other labels apply. The example is a good multiple choice question and could be used in an exam.",
    },
    {
        "label": "no_good_response",
        "keep": False,
        "description": "None of the responses are correct or otherwise of high quality.",
    },
    {
        "label": "ambiguous",
        "keep": False,
        "description": "It is ambiguous which response is best because multiple responses are correct and of high quality.",
    },
    {
        "label": "unintended_queues",
        "keep": False,
        "description": "The instruction or input contains unintended queues that make it easy to identify the correct response. For example, the correct response is much shorter than the others.",
    },
    {
        "label": "instruction_illposed",
        "keep": False,
        "description": "The instruction cannot be followed because it makes a false assumption or is a 'trick question.'",
    },
    {
        "label": "action_cannot_be_taken",
        "keep": False,
        "description": "The instruction cannot be followed by a language model because language models only output text.",
    },
    {
        "label": "includes_answer_in_instruction",
        "keep": False,
        "description": "The prompt includes the desired response or the correct answer is otherwise given away.",
    },
    {
        "label": "maybe_problematic",
        "keep": False,
        "description": "If there might be a problem with an example, err on the side of labeling it as 'maybe_problematic'.",
    },
]


def prepend_to_instruction(prefix, formats):
    for i, format in enumerate(formats):
        split = format.split("{instruction}")
        new_format = split[0] + prefix + "{instruction}" + split[1]
        formats[i] = new_format
    return formats


def set_fixed_instruction(instruction):
    format = general_formats[1]
    new_format = format.format(instruction=instruction, input="{instruction}")
    return [new_format]


@dataclass
class DistributionGenerator:
    name: str = None
    leading_sentence: str = None
    blacklisted_words: list = None
    requirements: list = None
    option_requirements = general_option_requirements
    options_demonstrations = None
    filter_labels = general_filter_labels
    headings = general_headings
    formats = general_formats
    examples: list = None
    batch_size: int = 20
    resample = True
    difficulty_values = ["easy", "medium", "hard"]
    instructions_check = lambda e: True
    options_check = lambda e: True
    primary_key = "instruction"
    post_process = lambda e: e
