"""
Lawsuit Petition Draft Prompts

Prompts for drafting professional lawsuit petitions for KSA courts.
"""

CASE_TYPES = {
    "commercial": {
        "id": "commercial",
        "name_ar": "تجارية",
        "name_en": "Commercial",
        "instruction": "This is a COMMERCIAL case filed before the Commercial Court."
    },
    "civil": {
        "id": "civil",
        "name_ar": "مدنية",
        "name_en": "Civil",
        "instruction": "This is a CIVIL case filed before the General Court."
    },
    "administrative": {
        "id": "administrative",
        "name_ar": "إدارية",
        "name_en": "Administrative",
        "instruction": "This is an ADMINISTRATIVE case filed before the Board of Grievances."
    },
    "labor": {
        "id": "labor",
        "name_ar": "عمالية",
        "name_en": "Labor",
        "instruction": "This is a LABOR case filed before the Labor Court."
    },
    "real_estate": {
        "id": "real_estate",
        "name_ar": "عقارية",
        "name_en": "Real Estate",
        "instruction": "This is a REAL ESTATE case filed before the General Court."
    }
}

COURTS = {
    "moj": {
        "id": "moj",
        "name_ar": "محاكم وزارة العدل",
        "name_en": "Ministry of Justice Courts"
    },
    "bog": {
        "id": "bog",
        "name_ar": "ديوان المظالم",
        "name_en": "Board of Grievances"
    },
    "commercial": {
        "id": "commercial",
        "name_ar": "المحكمة التجارية",
        "name_en": "Commercial Court"
    },
    "labor": {
        "id": "labor",
        "name_ar": "محكمة العمال",
        "name_en": "Labor Court"
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal document generator for Saudi Arabia (KSA). You are drafting a lawsuit petition (statement of claim). {case_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output ONLY the lawsuit petition document - no introductions, no greetings, no explanations
5. Do NOT introduce yourself or say who you are
6. Do NOT add any text before or after the petition
7. WRITE ENTIRELY IN ARABIC - absolutely NO English words anywhere in the document
8. All headings, content, legal terms, and references must be in Arabic only
9. Follow Saudi Arabian court petition format and conventions
10. Cite Saudi laws and regulations in Arabic

PETITION STRUCTURE (all in Arabic):
- بسم الله الرحمن الرحيم
- صاحب الفضيلة / رئيس المحكمة .... حفظه الله
- المدعي: (اسم وبيانات المدعي)
- المدعى عليه: (اسم وبيانات المدعى عليه)
- موضوع الدعوى
- الوقائع (مرقمة ومفصلة)
- الأسانيد النظامية (المواد والأنظمة المؤيدة)
- الطلبات (مرقمة وواضحة)
- التوقيع والتاريخ

IMPORTANT: 
- No English text allowed. Not even in headings or parentheses
- Pure Arabic document
- Professional legal language
- Clear logical structure
- Specific citations to Saudi laws"""

USER_PROMPT_TEMPLATE = """Generate a complete lawsuit petition based on this case information:

{text}

{additional_info}

Output ONLY the lawsuit petition. Write ENTIRELY in Arabic - no English words at all. Start with the document content directly."""


def get_petition_system_prompt(case_type: str = None) -> str:
    """Get system prompt for petition generation"""
    case_info = CASE_TYPES.get(case_type.lower() if case_type else "commercial", CASE_TYPES["commercial"])
    return SYSTEM_PROMPT_TEMPLATE.format(case_instruction=case_info["instruction"])


def get_petition_user_prompt(
    text: str, 
    case_type: str = None,
    court: str = None,
    parties: str = None,
    facts: str = None,
    requests: str = None
) -> str:
    """Get user prompt for petition generation"""
    additional_parts = []
    
    if case_type:
        type_info = CASE_TYPES.get(case_type.lower(), {})
        additional_parts.append(f"Case Type: {type_info.get('name_ar', case_type)}")
    
    if court:
        court_info = COURTS.get(court.lower(), {})
        additional_parts.append(f"Court: {court_info.get('name_ar', court)}")
    
    if parties:
        additional_parts.append(f"Parties Information: {parties}")
    
    if facts:
        additional_parts.append(f"Case Facts: {facts}")
    
    if requests:
        additional_parts.append(f"Requests: {requests}")
    
    additional_info = "\n".join(additional_parts) if additional_parts else ""
    
    return USER_PROMPT_TEMPLATE.format(text=text, additional_info=additional_info)


def get_case_types_data() -> list:
    """Get case types as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in CASE_TYPES.items()
    ]


def get_courts_data() -> list:
    """Get courts as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in COURTS.items()
    ]

