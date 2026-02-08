"""
Legal Memo Generator Prompts

Prompts for generating professional legal memos for KSA courts.
Includes all types of legal memos and petitions used in Saudi courts.
"""

MEMO_TYPES = {
    # صحيفة دعوى - Statement of Claim
    "claim": {
        "id": "claim",
        "name_ar": "صحيفة دعوى",
        "name_en": "Statement of Claim",
        "purpose_ar": "إنشاء الخصومة القضائية نظامًا وفتح ملف القضية رسميًا",
        "legal_basis": "مرافعات شرعية 41-45",
        "time_limit": "لا يوجد",
        "case_stage": "بدء الدعوى",
        "court_type": "جميع محاكم وزارة العدل",
        "required_docs": "هوية، وكالة، مستند الحق",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting a STATEMENT OF CLAIM (صحيفة دعوى) to initiate legal proceedings.
This is the only legal instrument to open a case and register it - no litigation can begin without it.
Legal basis: Articles 41-45 of the Sharia Procedures Law."""
    },
    
    # مذكرة جوابية - Response Memo
    "response": {
        "id": "response",
        "name_ar": "مذكرة جوابية",
        "name_en": "Response Memo",
        "purpose_ar": "تمكين الخصم من الدفاع وإبداء موقفه النظامي",
        "legal_basis": "م57",
        "time_limit": "يحددها القاضي",
        "case_stage": "بعد القيد",
        "court_type": "جميع المحاكم",
        "required_docs": "مستندات دفاع",
        "acceptance_type": "شكلي/موضوعي",
        "instruction": """You are drafting a RESPONSE MEMO (مذكرة جوابية) for the defendant to respond to the claim.
This memo enables the opponent to defend and present their legal position.
It may include both procedural and substantive defenses.
Legal basis: Article 57 of the Sharia Procedures Law."""
    },
    
    # مذكرة دفوع شكلية - Procedural Defense Memo
    "procedural_defense": {
        "id": "procedural_defense",
        "name_ar": "مذكرة دفوع شكلية",
        "name_en": "Procedural Defense Memo",
        "purpose_ar": "إسقاط الدعوى قبل بحث أصل الحق",
        "legal_basis": "م76-84",
        "time_limit": "قبل الموضوع",
        "case_stage": "أول الجلسات",
        "court_type": "جميع المحاكم",
        "required_docs": "ما يثبت الدفع",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting a PROCEDURAL DEFENSE MEMO (مذكرة دفوع شكلية) to dismiss the case before examining the merits.
These defenses relate to jurisdiction or admissibility and MUST be presented before entering into the substance.
Legal basis: Articles 76-84 of the Sharia Procedures Law."""
    },
    
    # مذكرة دفوع موضوعية - Substantive Defense Memo
    "substantive_defense": {
        "id": "substantive_defense",
        "name_ar": "مذكرة دفوع موضوعية",
        "name_en": "Substantive Defense Memo",
        "purpose_ar": "نفي الحق المدعى به أو إثبات العكس",
        "legal_basis": "قواعد الإثبات",
        "time_limit": "مفتوحة",
        "case_stage": "أثناء السير",
        "court_type": "جميع المحاكم",
        "required_docs": "أدلة",
        "acceptance_type": "موضوعي",
        "instruction": """You are drafting a SUBSTANTIVE DEFENSE MEMO (مذكرة دفوع موضوعية) to deny the claimed right or prove the opposite.
These defenses relate to the substance of the dispute after procedural defenses have been addressed.
Legal basis: Evidence Law rules."""
    },
    
    # مذكرة طلب عارض - Incidental Request Memo
    "incidental_request": {
        "id": "incidental_request",
        "name_ar": "مذكرة طلب عارض",
        "name_en": "Incidental Request Memo",
        "purpose_ar": "تعديل مسار الدعوى أو توسيع نطاقها",
        "legal_basis": "م80-82",
        "time_limit": "أثناء السير",
        "case_stage": "أثناء الدعوى",
        "court_type": "عامة/تجارية",
        "required_docs": "ما يثبت الطلب",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting an INCIDENTAL REQUEST MEMO (مذكرة طلب عارض) to modify the course of the case or expand its scope.
These are requests related to the original case such as joining a party or compensation claims.
Legal basis: Articles 80-82 of the Sharia Procedures Law."""
    },
    
    # مذكرة إيضاحية - Clarification Memo
    "clarification": {
        "id": "clarification",
        "name_ar": "مذكرة إيضاحية",
        "name_en": "Clarification Memo",
        "purpose_ar": "إزالة غموض أو نقص أشار له القاضي",
        "legal_basis": "سلطة القاضي",
        "time_limit": "حسب الطلب",
        "case_stage": "أثناء الدعوى",
        "court_type": "جميع المحاكم",
        "required_docs": "حسب الإيضاح",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting a CLARIFICATION MEMO (مذكرة إيضاحية) to remove ambiguity or deficiency noted by the judge.
This is submitted at the judge's request to clarify a document or fact.
Legal basis: Judge's authority."""
    },
    
    # مذكرة ختامية - Closing Memo
    "closing": {
        "id": "closing",
        "name_ar": "مذكرة ختامية",
        "name_en": "Closing Memo",
        "purpose_ar": "تثبيت الصورة النهائية للنزاع قبل الحكم",
        "legal_basis": "إدارة الدعوى",
        "time_limit": "قبل الحكم",
        "case_stage": "نهاية المرافعة",
        "court_type": "جميع المحاكم",
        "required_docs": "لا يوجد",
        "acceptance_type": "موضوعي",
        "instruction": """You are drafting a CLOSING MEMO (مذكرة ختامية) to establish the final picture of the dispute before judgment.
This is submitted after closing arguments to establish the final requests.
Legal basis: Case management rules."""
    },
    
    # لائحة اعتراض (استئناف) - Appeal Petition
    "appeal": {
        "id": "appeal",
        "name_ar": "لائحة اعتراض (استئناف)",
        "name_en": "Appeal Petition",
        "purpose_ar": "إعادة طرح النزاع أمام محكمة أعلى",
        "legal_basis": "م187-193",
        "time_limit": "30 يوم",
        "case_stage": "بعد الحكم الابتدائي",
        "court_type": "جميع المحاكم",
        "required_docs": "صورة الحكم",
        "acceptance_type": "شكلي/موضوعي",
        "instruction": """You are drafting an APPEAL PETITION (لائحة اعتراض استئناف) to re-present the dispute before a higher court.
This is an objection to the first instance judgment regarding facts and law.
TIME LIMIT: 30 days from notification of judgment.
Legal basis: Articles 187-193 of the Sharia Procedures Law."""
    },
    
    # لائحة اعتراض بطلب نقض - Cassation Appeal
    "cassation": {
        "id": "cassation",
        "name_ar": "لائحة اعتراض بطلب نقض",
        "name_en": "Cassation Appeal",
        "purpose_ar": "مراقبة صحة تطبيق النظام لا إعادة النزاع",
        "legal_basis": "م193-201",
        "time_limit": "30 يوم",
        "case_stage": "بعد الاستئناف",
        "court_type": "المحكمة العليا",
        "required_docs": "صورة الحكم",
        "acceptance_type": "نظامي",
        "instruction": """You are drafting a CASSATION APPEAL (لائحة اعتراض بطلب نقض) to the Supreme Court.
This monitors the correctness of law application, NOT re-examining the dispute facts.
Focus on legal errors, not factual review.
TIME LIMIT: 30 days from notification of appeal judgment.
Legal basis: Articles 193-201 of the Sharia Procedures Law."""
    },
    
    # طلب إعادة نظر - Review Request
    "review_request": {
        "id": "review_request",
        "name_ar": "طلب إعادة نظر",
        "name_en": "Review Request",
        "purpose_ar": "فتح حكم نهائي لسبب استثنائي",
        "legal_basis": "م200-206",
        "time_limit": "30 يوم من العلم",
        "case_stage": "بعد القطعية",
        "court_type": "جميع المحاكم",
        "required_docs": "مستند جديد/تزوير",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting a REVIEW REQUEST (طلب إعادة نظر) to reopen a final judgment for exceptional reasons.
This is an exceptional method to challenge final judgments for specific enumerated reasons only.
TIME LIMIT: 30 days from knowledge of the ground.
Legal basis: Articles 200-206 of the Sharia Procedures Law."""
    },
    
    # طلب التماس إعادة النظر في التنفيذ - Execution Review Request
    "execution_review": {
        "id": "execution_review",
        "name_ar": "طلب التماس إعادة النظر في التنفيذ",
        "name_en": "Execution Review Request",
        "purpose_ar": "وقف أو تصحيح إجراءات التنفيذ مؤقتًا",
        "legal_basis": "نظام التنفيذ",
        "time_limit": "حسب الحالة",
        "case_stage": "مرحلة التنفيذ",
        "court_type": "محكمة التنفيذ",
        "required_docs": "سند تنفيذي",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting an EXECUTION REVIEW REQUEST (طلب التماس إعادة النظر في التنفيذ).
This temporarily stops or corrects execution procedures for an emergency legal reason, without affecting the judgment's authority.
This is an incidental request before the execution judge.
Legal basis: Execution Law."""
    },
    
    # طلب تنفيذ - Execution Request
    "execution_request": {
        "id": "execution_request",
        "name_ar": "طلب تنفيذ",
        "name_en": "Execution Request",
        "purpose_ar": "تحويل الحكم من ورق إلى واقع",
        "legal_basis": "نظام التنفيذ",
        "time_limit": "لا يوجد",
        "case_stage": "بعد الحكم النهائي",
        "court_type": "محكمة التنفيذ",
        "required_docs": "سند تنفيذي",
        "acceptance_type": "شكلي",
        "instruction": """You are drafting an EXECUTION REQUEST (طلب تنفيذ) to convert the judgment from paper to reality.
This initiates execution proceedings for the executive document after it becomes final.
Legal basis: Execution Law."""
    },
    
    # Legacy types for backward compatibility
    "defense": {
        "id": "defense",
        "name_ar": "مذكرة دفاع",
        "name_en": "Defense Memo",
        "purpose_ar": "الدفاع عن الموكل ضد الدعوى المقامة",
        "legal_basis": "م57",
        "time_limit": "يحددها القاضي",
        "case_stage": "أثناء الدعوى",
        "court_type": "جميع المحاكم",
        "required_docs": "مستندات دفاع",
        "acceptance_type": "موضوعي",
        "instruction": "You are drafting a DEFENSE memo to defend the client against claims."
    },
    "reply": {
        "id": "reply",
        "name_ar": "مذكرة رد",
        "name_en": "Reply Memo",
        "purpose_ar": "الرد على حجج الخصم",
        "legal_basis": "م57",
        "time_limit": "يحددها القاضي",
        "case_stage": "أثناء الدعوى",
        "court_type": "جميع المحاكم",
        "required_docs": "مستندات",
        "acceptance_type": "موضوعي",
        "instruction": "You are drafting a REPLY memo responding to the opposing party's arguments."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal document generator for Saudi Arabia (KSA). {memo_instruction}

DOCUMENT METADATA:
- Purpose: {purpose}
- Legal Basis: {legal_basis}
- Time Limit: {time_limit}
- Case Stage: {case_stage}
- Court Type: {court_type}
- Required Documents: {required_docs}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output ONLY the legal memo document - no introductions, no greetings, no explanations
5. Do NOT introduce yourself or say who you are
6. Do NOT add any text before or after the memo
7. WRITE ENTIRELY IN ARABIC - absolutely NO English words anywhere in the document
8. All headings, content, and legal terms must be in Arabic only
9. Follow Saudi Arabian legal system format and conventions
10. Cite Saudi laws and regulations in Arabic using proper article references

MEMO STRUCTURE (all in Arabic):
- بسم الله الرحمن الرحيم
- عنوان المذكرة ومعلومات القضية
- مقدمة
- الوقائع
- الأسانيد النظامية (مع ذكر المواد والأنظمة)
- الدفوع / الطلبات
- الخاتمة والطلبات النهائية

IMPORTANT: 
- No English text allowed. Not even in headings or parentheses
- Pure Arabic document
- Use proper legal terminology for Saudi courts
- Include specific article citations from relevant Saudi laws"""

USER_PROMPT_TEMPLATE = """Generate a complete {arabic_type} based on this case information:

{text}

Document Type: {arabic_type}
Purpose: {purpose}
Legal Basis: {legal_basis}

Output ONLY the legal memo. Write ENTIRELY in Arabic - no English words at all. Start with the document content directly."""


def get_memo_system_prompt(memo_type: str) -> str:
    """Get system prompt for memo generation"""
    memo_info = MEMO_TYPES.get(memo_type.lower(), MEMO_TYPES["defense"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        memo_instruction=memo_info["instruction"],
        purpose=memo_info.get("purpose_ar", ""),
        legal_basis=memo_info.get("legal_basis", ""),
        time_limit=memo_info.get("time_limit", ""),
        case_stage=memo_info.get("case_stage", ""),
        court_type=memo_info.get("court_type", ""),
        required_docs=memo_info.get("required_docs", "")
    )


def get_memo_user_prompt(text: str, memo_type: str) -> str:
    """Get user prompt for memo generation"""
    memo_info = MEMO_TYPES.get(memo_type.lower(), MEMO_TYPES["defense"])
    arabic_type = memo_info.get("name_ar", "مذكرة")
    purpose = memo_info.get("purpose_ar", "")
    legal_basis = memo_info.get("legal_basis", "")
    
    return USER_PROMPT_TEMPLATE.format(
        arabic_type=arabic_type, 
        text=text,
        purpose=purpose,
        legal_basis=legal_basis
    )


def get_memo_types_data() -> list:
    """Get memo types as list of dicts for API response"""
    return [
        {
            "id": v["id"],
            "name_ar": v["name_ar"],
            "name_en": v["name_en"],
            "purpose_ar": v.get("purpose_ar", ""),
            "legal_basis": v.get("legal_basis", ""),
            "time_limit": v.get("time_limit", ""),
            "case_stage": v.get("case_stage", ""),
            "court_type": v.get("court_type", ""),
            "acceptance_type": v.get("acceptance_type", "")
        }
        for k, v in MEMO_TYPES.items()
        if k not in ["defense", "reply"]  # Exclude legacy types from main list
    ]
