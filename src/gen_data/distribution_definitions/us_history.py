from dataclasses import dataclass
from gen_data.gen_utils.distribution_generator import (
    DistributionGenerator,
    general_sentence,
    set_fixed_instruction,
)


class USHistory(DistributionGenerator):
    name = "us_history"
    leading_sentence = (
        general_sentence
        + " Each instruction should request that the GPT model answer a question about U.S. history."
    )

    requirements = [
        "The instructions should be in English.",
        "Under the 'difficulty_to_grade' heading, label instructions as 'easy,' 'medium,' or 'hard.'",
        "You must alternate between the difficulty levels.", 
        "The easy questions should be answerable by the average U.S. citizen.",
        "The hard questions should lean more towards history synthesis and understand and less toward fact recall.",
    ]

    formats = set_fixed_instruction("Provide an accurate response to the following history question.")

    examples = [
        {
            "instruction": "What was the name of the first president of the United States?",
            "preferred_completion": "George Washington",
            "difficulty_to_grade": "easy"
        },
        {
            "instruction": "What war was fought between the North and the South in the United States?",
            "preferred_completion": "Civil War",
            "difficulty_to_grade": "easy"
        },
        {
            "instruction": "Which amendment to the U.S. Constitution abolished slavery?",
            "preferred_completion": "13th Amendment",
            "difficulty_to_grade": "easy"
        },
        {
            "instruction": "Which document serves as the supreme law of the United States?",
            "preferred_completion": "The Constitution",
            "difficulty_to_grade": "easy"
        },
        {
            "instruction": "Who delivered the famous 'I Have a Dream' speech during the Civil Rights March on Washington in 1963?",
            "preferred_completion": "Martin Luther King Jr.",
            "difficulty_to_grade": "easy"
        },
        {
            "instruction": "In the early 20th century, the Progressive Era marked a period of reform in the United States. What were some of the key social and political issues that Progressives aimed to address?",
            "preferred_completion": "Issues such as women's suffrage, workers' rights, anti-monopoly regulations, and political corruption",
            "difficulty_to_grade": "medium"
        },
        {
            "instruction": "During the mid-1800s, the concept of 'Manifest Destiny' played a significant role in American expansion. What did 'Manifest Destiny' represent, and how did it impact U.S. territorial expansion?",
            "preferred_completion": "'Manifest Destiny' was the belief that Americans were destined to expand across the continent from the Atlantic to the Pacific. It justified westward expansion and the acquisition of new territories.",
            "difficulty_to_grade": "medium"
        },
        {
            "instruction": "The Great Depression of the 1930s had a profound impact on American society. What were some of the major economic and social consequences of the Great Depression, and how did the government respond to address these challenges?",
            "preferred_completion": "Consequences included widespread unemployment, bank failures, and economic hardship. The government responded with programs like the New Deal, which included social welfare measures and public works projects.",
            "difficulty_to_grade": "medium"
        },
        {
            "instruction": "During the American Civil Rights Movement of the 1950s and 1960s, several key figures emerged as leaders. Who were some of these prominent leaders, and what were their contributions to the movement?",
            "preferred_completion": "Prominent leaders included Martin Luther King Jr., Rosa Parks, Malcolm X, and others. They played crucial roles in advocating for racial equality and civil rights through nonviolent protests, legal actions, and speeches.",
            "difficulty_to_grade": "medium"
        },
        {
            "instruction": "This question refers to the following information.\n\"I was once a tool of oppression\nAnd as green as a sucker could be\nAnd monopolies banded together\nTo beat a poor hayseed like me.\n\"The railroads and old party bosses\nTogether did sweetly agree;\nAnd they thought there would be little trouble\nIn working a hayseed like me. . . .\"\n\u2014\"The Hayseed\"\nWhich of the following is an accomplishment of the political movement that was organized around sentiments similar to the one in the song lyrics above?",
            "preferred_completion": "Enactment of laws regulating railroads.",
            "difficulty_to_grade": "hard"
        },
        {
            "instruction": "This question refers to the following information.\n\"The power . . . given to the commanding officer over all the people of each district is that of an absolute monarch. His mere will is to take the place of all law. . . . It reduces the whole population of the ten states\u2014all persons, of every color, sex, and condition, and every stranger within their limits\u2014to the most abject and degrading slavery.\"\nThe political sentiment of the veto message above is most similar to which of the following political positions taken in the twentieth century?",
            "preferred_completion": "Governor Orval Faubus's response to the steps taken by President Dwight Eisenhower to resolve the Little Rock crisis in 1957.",
            "difficulty_to_grade": "hard"
        },
        {
            "instruction": "This question refers to the following information.\n\"We conclude that, in the field of public education, the doctrine of \"separate but equal\" has no place. Separate educational facilities are inherently unequal. Therefore, we hold that the plaintiffs and others similarly situated for whom the actions have been brought are, by reason of the segregation complained of, deprived of the equal protection of the laws guaranteed by the Fourteenth Amendment.\"\nBrown v. Board of Education, 1954\nWhich of the following best represents an effect of the legal decision described above?",
            "preferred_completion": "Continuing white resistance slowed efforts at desegregation, sparking a series of social conflicts throughout the South.",
            "difficulty_to_grade": "hard"
        },
        {
            "instruction": "This question refers to the following information.\n\"The seeds of totalitarian regimes are nurtured by misery and want. They spread and grow in the evil soil of poverty and strife. They reach their full growth when the hope of a people for a better life has died. We must keep that hope alive. . . . Great responsibilities have been placed upon us by the swift movement of events. . . . I am confident that the Congress will face these responsibilities squarely.\"\n\u2014President Harry S. Truman, 1947\nThe passage above is part of President Truman's argument to Congress in favor of",
            "preferred_completion": "an extension of aid to Greece and Turkey.",
            "difficulty_to_grade": "hard"
        },
        {
            "instruction": "This question refers to the following information.\nOn Being Brought from Africa to America\n'Twas mercy brought me from my Pagan land,\nTaught my benighted soul to understand\nThat there's a God, that there's a Saviour too;\nOnce I redemption neither sought nor knew.\nSome view our sable race with scornful eye,\n\"Their colour is a diabolic die.\"\nRemember, Christians, Negroes, black as Cain,\nMay be refin'd, and join th' angelic train.\n\u2014Phillis Wheatley, Poems on Various Subjects, Religious and Moral, 1773\nThe point of Wheatley's poem can best be compared to which of the following?",
            "preferred_completion": "Martin Luther King, Jr.'s \"I Have a Dream\u201d speech",
            "difficulty_to_grade": "hard"
        },
        {
            "instruction": "This question refers to the following information.\n\"We are men; we have souls, we have passions, we have feelings, we have hopes, we have desires, like any other race in the world. The cry is raised all over the world today of Canada for the Canadians, of America for the Americans, of England for the English, of France for the French, of Germany for the Germans\u2014do you think it is unreasonable that we, the Blacks of the world, should raise the cry of Africa for the Africans?\"\n\u2014Marcus Garvey, 1920\nThe passage above presents a position in which of the following ongoing debates in American history?",
            "preferred_completion": "The debate between separatism and integration when it came to the place of African Americans in American society.",
            "difficulty_to_grade": "hard"
        }
    ]
    headings = [
        {"key": "instruction", "heading": "Instruction"},
        {"key": "preferred_completion", "heading": "Output"},
        {"key": "difficulty_to_grade", "heading": "Difficulty to Grade"},
    ]
    option_requirements = [
        "Make the multiple choice options diverse.", 
        "Give answers that would look correct to a student but are actually wrong. The wrong answers should include very plausible made up facts or low-quality analyses.",
        "Don't make the worse answers sound ridiculous.",
        "Don't make the worse responses shorter than the good response."
    ]

    options_demonstrations = [{
        "instruction": "What was the name of the first president of the United States?",
        "responses": {
            "George Washington": 1,
            "Thomas Jefferson": 0,
            "John Adams": 0,
            "James Madison": 0
        },
    },
    {
        "instruction": "In what year was the Declaration of Independence signed?",
        "responses": {
            "1776": 1,
            "1775": 0,
            "1777": 0,
            "1778": 0
        },
    },
    {
        "instruction": "This question refers to the following information.\n\"The power . . . given to the commanding officer over all the people of each district is that of an absolute monarch. His mere will is to take the place of all law. . . . It reduces the whole population of the ten states\u2014all persons, of every color, sex, and condition, and every stranger within their limits\u2014to the most abject and degrading slavery.\"\nThe political sentiment of the veto message above is most similar to which of the following political positions taken in the twentieth century?",
        "responses": {
            "Justice Frank Murphy's dissent in the Supreme Court case, Korematsu v. United States in 1944.": 0.0,
            "U.S. Army lawyer Joseph Welsh's opposition to Senator Joseph McCarthy in the Army-McCarthy hearings in 1954.": 0.0,
            "Governor Orval Faubus's response to the steps taken by President Dwight Eisenhower to resolve the Little Rock crisis in 1957.": 1.0,
            "John Lewis's endorsement of the Voting Rights Act in 1965.": 0.0
        }
    }
    ]