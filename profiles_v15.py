# ===========================================================
# profiles_v15.py
# Dynamic Profile Configurations for StoryGrid v5.0 (Production-Ready)
# V15.1: Stance vs. Modifier Posture Model implementation.
# V15.5: Prop-Aware Filtering support (Lexicon refinement).
# ===========================================================

# Emotion Conflict Groups
EMOTION_CONFLICT_GROUPS = [
    {
        "name": "Joy/Sadness Axis", 
        "emotions": ["vui vẻ", "hạnh phúc", "tự hào", "háo hức", "hào hứng", "buồn bã", "thất vọng", "hối hận", "ái ngại", "lí nhí"]
    },
    {
        "name": "Calm/Anger Axis",
        "emotions": ["bình tĩnh", "khoan thai", "nhẹ nhàng", "tức giận", "cáu kỉnh", "quyết tâm", "tự tin"]
    },
]

# V15.1: Stance Conflict Map (Actions that break a stance)
STANCE_CONFLICT_MAP = {
    "ngồi": ["lao đi", "chạy", "nhảy", "đứng dậy", "bước tới", "đi bộ", "lật đật chạy", "chạy vòng quanh", "nhún nhảy", "xúm lại", "vươn vai"],
    "nằm": ["lao đi", "chạy", "nhảy", "đứng dậy", "ngồi dậy", "bước tới", "đi bộ", "xúm lại"],
}

# V15.1: Modifier Conflict Map (Actions that break a modifier)
MODIFIER_CONFLICT_MAP = {
    "khoanh tay": ["giơ tay", "vồ ngay lấy", "cầm", "ôm", "chỉ vào", "vật lộn ôm", "kéo tấm khăn", "phủi đất", "vỗ vỗ", "gãi bụng", "chỉnh lại nơ"],
    "quay lưng lại": ["quay mặt lại", "ôm chầm lấy"],
}


# Define the default profile (Pixar 3D, Vietnamese)
PIXAR_3D_VI_PROFILE = {
    "language": "vi",
    "genre": "pixar_3d_animation",
    "StoryGrid_Version": "5.0", # V15 Update
    "parsing_rules": {
        "default_passive_action": "quan sát",
        "default_passive_emotion": None, 
        "default_passive_stance": None, # V15.1

        "attribute_negations": {
            "phủi đất": ["dính đầy đất", "lấm lem"],
            "lau khô": ["ướt sũng"],
        },
        
        "emotion_conflict_groups": EMOTION_CONFLICT_GROUPS,
        "stance_conflict_map": STANCE_CONFLICT_MAP, # V15.1
        "modifier_conflict_map": MODIFIER_CONFLICT_MAP, # V15.1

        "scene_header_patterns": [
            r"^\s*(?:[\-\*\•#]+\s*)?(?:CẢNH|Cảnh)\s+(\d+|MỞ\s*ĐẦU|KẾT)\s*[:\-]?\s+.*$",
        ],
        "structure_labels": {
            "Bối cảnh": "setting", "Hành động": "action", "Sự xuất hiện": "arrival",
            "Cao trào mở đầu": "climax", "Cao trào": "climax", "Kết": "conclusion"
        },
        "character_blacklist": [
            "Cảnh", "Scene", "VÚT", "ĐÂY", "Màu", "Hả", "Ồ",
            "Sẵn sàng", "Bắt đầu",
            "CHÍNH XÁC", "KHỔNG LỒ", "MÁT LẠNH", "TUYỆT VỜI", "SỨC MẠNH CỦA MÀU SẮC",
        ],
        "group_agents": ["Ba bạn nhỏ", "Cả lớp", "Các bạn"],
        
        # V15.2: Contextual references (Anaphora)
        "contextual_references": {
            "cô bé": "female_child",
            "cậu bé": "male_child",
            "thầy": "teacher",
            "cô": "teacher",
        },

        "invalid_name_suffixes": ["đang", "đặt", "chạy", "ngồi", "bước", "kéo", "nhìn", "không", "chậm rãi", "khoan thai", "bước tới"],
        "invalid_name_prefixes": ["ăn", "bảo", "chọn", "giúp", "nhưng", "phải", "nếu", "con đoán"],
        "role_self_teacher": ["thầy","cô"],
        
        "negation_keywords": ["không", "chưa", "đừng", "ngừng"],
        
        "upper_chars": "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ",
        "lower_chars": "a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ",
    },
    "lexicons": {
        "appearance_phrases": [
            # Inherent/Static
            "Khăn quàng xanh", "Nơ hồng", "Xinh đẹp",
            # Colors/Adjectives
            "xanh", "đỏ", "vàng", "hồng", "tím", "cam", "đen", "trắng", "nâu",
            "xinh", "đẹp",
            # Dynamic/Temporary (Physical states)
            "dính đầy đất", "lấm lem", "ướt sũng",
            "run run", "bay phấp phới",
            # Descriptors
            "rực rỡ", "long lanh",
        ],
        
        # V15.1: Action Classification Refinement
        "dynamic_actions": [
            # High-Intensity
            "lao đi", "vồ ngay lấy", "ném xuống đất", "lật đật chạy", "vật lộn ôm",
            "giơ tay", "dậm dậm chân", "hét lớn",
            # Medium-Intensity
            "chạy vòng quanh", "xúm lại", "kéo tấm khăn", "nhún nhảy", "vươn vai",
            "bước tới", "đặt chiếc giỏ", "cầm lên", "nhặt lên", "phủi đất", "ôm chầm lấy", "trang trí", "quây quần",
            "chia cho", "cắn thử", "dừng lại", "đứng dậy", "bước đến", "chạy đến",
            # Low-Intensity/Micro-actions
            "hít hít mũi", "gãi bụng", "chỉnh lại nơ", "thở hổn hển", "đi bộ",
            "soi mình", "chỉ vào", "cầm", "nhìn", "ngước nhìn", "hít một hơi", "vỗ vỗ",
            # Communication/Expression Actions
            "cười hiền", "mỉm cười", "cười lớn", "cười", "nói", "nói rất nhanh", "đồng thanh",
            
            # States as Actions
            "ngáp ngắn ngáp dài", "nảy mầm",
            # Passive
            "quan sát", "lắng nghe", "thấy",
            # Adverbs/Dynamic States
            "chậm rãi", "khoan thai",
        ],
        
        # V15.1: Stances (Mutually Exclusive)
        "stances": [
            "đứng", "ngồi", "nằm",
        ],
        
        # V15.1: Posture Modifiers
        "posture_modifiers": [
             "khoanh tay", "quay lưng lại",
        ],

        "emotion_phrases": [
            "háo hức", "tò mò", "hào hứng", "bất ngờ", "quyết tâm", "tự tin",
            "tức giận", "ái ngại", "buồn bã", "thất vọng", "hối hận",
            "vui vẻ", "hạnh phúc", "tự hào",
            "nhỏ nhẹ", "gật gù", "lí nhí", "suy nghĩ",
            "bí ẩn", "tuyệt vời",
            # Emotional/Physical Expressions
            "mắt tròn xoe", "tai cụp xuống", "đỏ mặt", "mắt sáng rỡ",
        ],
        "props_list": [
            "giỏ mây","bảng màu","hoa","sách","bút",
            "khăn lanh","phiến đá","búp măng", "tấm khăn lanh màu be",
            "gốc cây cổ thụ", "vũng nước", "chiếc giỏ",
            # Generics
            "khăn", "nơ", "áo", "mũ",
            # Vegetables/Fruits
            "cà chua","cà rốt","súp lơ","cà tím","ớt chuông vàng","củ dền","dưa chuột","bông cải xanh",
            "táo","chuối","chanh",
            # Specific variations
            "hoa cúc ánh dương",
            "quả cà chua", "củ cà rốt", "quả táo", "quả ớt chuông vàng", "rau củ", "cây Súp lơ xanh",
            "quả dưa chuột",
        ],
        "tone_map": [
            ("ấm áp","warm"),("tò mò","curious"),("nhẹ nhàng","gentle"),
            ("hào hứng","excited"), ("bí ẩn", "mysterious"), ("căng thẳng", "tense")
        ]
    },
    "cinematic_instructions": {
        "camera_moves": {
            "Slow-motion": "slow_motion",
            "Close-up": "close_up_shot",
            "Zoom": "zoom_in",
            "Fade": "fade_transition",
            "hài hước": "comedic_timing",
        },
        "vfx_sfx": {
            "bụi bay mù mịt": "dust_cloud_vfx",
            "lấp lánh": "sparkle_vfx",
            "tiếng nhạc": "background_music",
        },
        "meta_types": {
            r"Hình ảnh tưởng tượng": "visual_insert",
            r"Hình ảnh:": "visual_cue",
            r"Ghi chú:": "director_note",
            r"\(Lồng tiếng\)": "voice_over",
            r"\(V.O\)": "voice_over",
            r"\(V\.O\.\)": "voice_over",
        }
    }
}

# Function to load the active profile
def load_profile(profile_name="PIXAR_3D_VI"):
    # V15.1: Consolidate action/posture phrases for backward compatibility (e.g., classify_line)
    profile = PIXAR_3D_VI_PROFILE
    if 'action_phrases' not in profile['lexicons']:
        profile['lexicons']['action_phrases'] = profile['lexicons'].get('dynamic_actions', []) + \
                                                profile['lexicons'].get('stances', []) + \
                                                profile['lexicons'].get('posture_modifiers', [])
    return profile