from sec_memo_agents.core.templates import list_templates, load_template


def test_loads_six_workflow_templates():
    templates = list_templates()
    names = {template.name for template in templates}

    assert len(templates) >= 6
    assert "deal_screening" in names
    assert "credit_memo" in names
    assert load_template("due_diligence").required_sections
