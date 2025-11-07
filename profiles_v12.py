# ===========================================================
# profiles_v12.py
# Dynamic Profile Configurations for StoryGrid v4.0
# V12.1: Emotion Conflict Groups definition.
# V12.6: Lexicon Refinement (Improved classification).
# ===========================================================

# V12.1: Emotion Conflict Groups
# Defines groups where emotions are mutually exclusive.
EMOTION_CONFLICT_GROUPS = [
    {
        "name": "Joy/Sadness Axis", 
        "emotions": ["vui vẻ", "hạnh phúc", "tự hào", "háo hức", "hào hứng", "long lanh", "buồn bã", "thất vọng", "hối hận", "ái ngại"]
    },
    {
        "name": "Calm/Anger Axis",
        "emotions": ["bình tĩnh", "khoan thai", "nhẹ nhàng", "tức giận", "cáu kỉnh"]
    },
    # "Bất ngờ", "Bí ẩn", "Tò mò" are often transitional and may not conflict strongly.
]


# Define the default profile (Pixar 3D, Vietnamese)
PIXAR_3D_VI_PROFILE = {
    "language": "vi",
    "genre": "pixar_3d_animation",
    "StoryGrid_Version": "4.0", # V12 Update
    "parsing_rules": {
        "default_passive_action": "quan sát",
        "default_passive_emotion": None, 

        "attribute_negations": {
            "phủi đất": ["dính đầy đất", "lấm lem"],
            "lau khô": ["ướt sũng"],
        },
        
        # V12.1: Add conflict groups to rules
        "emotion_conflict_groups": EMOTION_CONFLICT_GROUPS,

        "scene_header_patterns": [
            r"^\s*(?:[\-\*\•#]+\s*)?(?:CẢNH|Cảnh)\s+(\d+|MỞ\s*ĐẦU|KẾT)\s*[:\-]?\s+.*$",
        ],
        "structure_labels": {
            "Bối cảnh": "setting", "Hành động": "action", "Sự xuất hiện": "arrival",
            "Cao trào mở đầu": "climax", "Cao trào": "climax", "Kết": "conclusion"
        },
        "character_blacklist": [
            "Cảnh", "Scene", "VÚT", "ĐÂY", "Màu", "Hả", "Ồ",
            "Sẵn sàng", "Bắt đầu", "CẢNH MỞ ĐẦU",
            "CHÍNH XÁC", "KHỔNG LỒ", "MÁT LẠNH", "TUYỆT VỜI", "SỨC MẠNH CỦA MÀU SẮC",
        ],
        "group_agents": ["Ba bạn nhỏ", "Cả lớp", "Các bạn"],

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
            "khăn", "nơ", "mũ", "kính", "áo", "quàng",
            "xanh", "đỏ", "vàng", "hồng", "tím", "cam", "đen", "trắng", "nâu",
            "xinh", "đẹp",
            # Dynamic/Temporary (Physical states)
            "dính đầy đất", "lấm lem", "ướt sũng",
            # V12.6: Moved dynamic/emotional states out: "mắt tròn xoe", "tai cụp xuống", "đỏ mặt", "bay phấp phới", "rực rỡ"
        ],
        "action_phrases": [
            # High-Intensity
            "lao đi", "vồ ngay lấy", "ném xuống đất", "lật đật chạy", "vật lộn ôm",
            "giơ tay", "dậm dậm chân", "hét lớn",
            # Medium-Intensity
            "chạy vòng quanh", "xúm lại", "kéo tấm khăn", "nhún nhảy", "vươn vai",
            "bước tới", "đặt chiếc giỏ", "cầm lên", "quay lưng lại", "khoanh tay",
            "bước đến", "nhặt lên", "phủi đất", "ôm chầm lấy", "trang trí", "quây quần",
            "chia cho", "cắn thử", "dừng lại", "đứng dậy",
            # Low-Intensity/Micro-actions
            "hít hít mũi", "gãi bụng", "chỉnh lại nơ", "thở hổn hển", "đi bộ",
            "soi mình", "chỉ vào", "cầm", "nhìn", "ngước nhìn", "hít một hơi",
            "cười hiền", "mỉm cười", "cười lớn",
            # States as Actions
            "ngáp ngắn ngáp dài", "nảy mầm",
            # Locomotion/Stance
            "đứng", "ngồi", "nằm",
            # Passive
            "quan sát", "lắng nghe",
            # Adverbs/Dynamic States (V12.6 Update)
            "chậm rãi", "khoan thai", "bay phấp phới",
        ],
        "emotion_phrases": [
            "háo hức", "tò mò", "hào hứng", "bất ngờ", "quyết tâm",
            "tức giận", "ái ngại", "buồn bã", "thất vọng", "hối hận",
            "vui vẻ", "hạnh phúc", "tự hào",
            "nhỏ nhẹ", "gật gù", "lí nhí", "suy nghĩ",
            "bí ẩn", "long lanh", "rực rỡ", "tuyệt vời",
            # Emotional/Physical Expressions (V12.6 Update)
            "mắt tròn xoe", "tai cụp xuống", "đỏ mặt",
        ],
        "props_list": [
            "giỏ mây","bảng màu","hoa","sách","bút",
            "khăn lanh","phiến đá","búp măng", "tấm khăn lanh màu be",
            "gốc cây cổ thụ", "vũng nước",
            # Vegetables/Fruits
            "cà chua","cà rốt","súp lơ","cà tím","ớt chuông vàng","củ dền","dưa chuột","bông cải xanh",
            "táo","chuối","chanh",
            # Specific variations
            "hoa cúc ánh dương", "chiếc giỏ",
            "quả cà chua", "củ cà rốt", "quả táo", "quả ớt chuông vàng", "rau củ", "cây Súp lơ xanh",
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
            r"Hình ảnh:": "visual_cue",
            r"Ghi chú:": "director_note",
            r"\(Lồng tiếng\)": "voice_over",
            r"\(V.O\)": "voice_over",
        }
    }
}

# Function to load the active profile
def load_profile(profile_name="PIXAR_3D_VI"):
    return PIXAR_3D_VI_PROFILE