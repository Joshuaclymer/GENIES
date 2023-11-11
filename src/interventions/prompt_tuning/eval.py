import fire

from interventions.classify_lora.eval import main


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs, intervention_name="prompt_tuning")


if __name__ == "__main__":
    fire.Fire(fire_wrap)
