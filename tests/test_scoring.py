from brltok.scoring import choice_correct, gold_choice, selected_choice

CH = ["A) bank", "B) library", "C) department store", "D) a lot of money", "E) new york"]


def test_text_answers():
    assert selected_choice("Reasoning...\nThe answer is: library", CH) == "B"
    assert selected_choice("The answer is: **department store**.", CH) == "C"
    assert selected_choice("The answer is: A lot of money", CH) == "D"   # article, not letter A
    assert selected_choice("The answer is: New York.", CH) == "E"


def test_letter_answers():
    assert selected_choice("The answer is: B", CH) == "B"
    assert selected_choice("The answer is: B) library", CH) == "B"
    assert selected_choice("The answer is: (C)", CH) == "C"
    assert selected_choice("The answer is: E. new york", CH) == "E"
    assert selected_choice("D", CH) == "D"


def test_braille_echo():
    assert selected_choice('The answer is: ,b"> library', CH) == "B"
    assert selected_choice('The answer is: a"> bank', CH) == "A"


def test_no_selection():
    assert selected_choice("The answer is: museum", CH) is None
    assert selected_choice("", CH) is None
    assert selected_choice("The answer is: bank or library", CH) is None


def test_last_marker_and_thinking():
    r = "<think>The answer is: bank? no.</think>\nThe answer is: library"
    assert selected_choice(r, CH) == "B"
    assert selected_choice("The answer is: bank. Wait.\nThe answer is: library", CH) == "B"


def test_choice_correct():
    assert gold_choice(["library"], CH) == "B"
    assert choice_correct("The answer is: B", CH, ["library"]) == 1.0
    assert choice_correct("The answer is: bank", CH, ["library"]) == 0.0
    # the released exact-match scorer gives 0 here; this one gives 1
    assert choice_correct("The answer is: C) department store", CH, ["department store"]) == 1.0


def test_letter_text_conflict_text_wins():
    ch = ["A) eating food", "B) public place", "C) race track", "D) at hotel", "E) public building"]
    assert selected_choice("The answer is: B) race track", ch) == "C"
    assert choice_correct("The answer is: B) race track", ch, ["race track"]) == 1.0
    assert choice_correct("The answer is: B", ch, ["race track"]) == 0.0


def test_duplicate_choice_text():
    ch = ["A) healthy", "B) passing around", "C) cure", "D) wellness", "E) healthy"]
    assert choice_correct("The answer is: A) healthy", ch, ["healthy"]) == 1.0
    assert choice_correct("The answer is: E", ch, ["healthy"]) == 1.0
    assert choice_correct("The answer is: healthy", ch, ["healthy"]) == 1.0
    assert choice_correct("The answer is: D", ch, ["healthy"]) == 0.0


def test_trailing_explanation():
    ch = ["A) forest", "B) street", "C) air", "D) caused by humans", "E) car show"]
    assert selected_choice("The answer is: the air. The other choices are not relevant.", ch) == "C"
    assert selected_choice("The answer is: C) air.  The word air fits.", ch) == "C"
    assert selected_choice("The answer is: E. car show", ch) == "E"
