"""Unit tests for modules.response_layer's internal-name leak guard."""

import pytest

from modules.response_layer import InternalLeakError, assert_no_internal_leakage


def test_clean_arabic_answer_passes():
    assert_no_internal_leakage("يوجد 8 مشاريع جارية في إدارة المشاريع المتخصصة.")


def test_clean_english_answer_passes():
    assert_no_internal_leakage("There are 8 Ongoing projects in the Specialized Projects department.")


def test_identifier_shaped_column_name_is_caught():
    with pytest.raises(InternalLeakError):
        assert_no_internal_leakage("هامش الربح محسوب من profit_pct لهذا المشروع.")


def test_raw_dept_literal_is_caught():
    with pytest.raises(InternalLeakError):
        assert_no_internal_leakage("المشروع تابع لإدارة BPO-Specialized Pr.")


def test_raw_status_literal_leaks_in_arabic_but_not_english():
    with pytest.raises(InternalLeakError):
        assert_no_internal_leakage("حالة المشروع هي Ongoing حالياً.")
    # Same literal word is legitimate in an English-language answer.
    assert_no_internal_leakage("The project status is Ongoing.")


def test_empty_text_is_a_noop():
    assert_no_internal_leakage("") is None
    assert_no_internal_leakage(None) is None
