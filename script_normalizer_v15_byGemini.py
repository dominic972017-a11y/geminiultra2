# ===========================================================
# script_normalizer_v15_byGemini.py
# Screenplay Normalizer → StoryGrid v5.0 (Production-Ready)
# - V15.0: Inter-Scene Persistence Architecture.
# - V15.1: Stance vs. Modifier Posture Model (Persistence 3.0).
# - V15.2: Local Contextual Anaphora Resolution.
# - V15.3: Single-Agent Association Fallback.
# - V15.4: Implicit Dialogue Recognition Engine.
# - V15.5: Prop-Aware Attribute Filtering.
# ===========================================================
import re, json, argparse, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
import itertools
import sys
import copy

# Import the profile loader (Updated to V15)
try:
    import profiles_v15 as profiles
except ImportError:
    print("❌ Lỗi: Không tìm thấy module 'profiles_v15.py'. Vui lòng đảm bảo file tồn tại.")
    sys.exit(1)


# ===========================================================
# SECTION 1: UTILITIES & HELPERS (Stable)
# ===========================================================

# (Utilities remain identical)
def nfc(s:str)->str: return unicodedata.normalize("NFC", s or "")

def canonicalize(text:str)->str:
    s = nfc(text)
    s = s.replace("\u00A0"," ").replace("\u2007"," ").replace("\u202F"," ")
    s = s.replace("：",":").replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-")
    s = s.replace("“",'"').replace("”",'"').replace("’","'")
    return s

def strip_combining(s:str)->str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def slugify(name:str)->str:
    s = strip_combining(name).lower()
    s = re.sub(r"[^a-z0-9]+","_", s).strip("_")
    s = s.replace("đ", "d")
    return s or "unnamed"

def clean_title(title:str)->str:
    t = title.strip()
    t = re.sub(r"^\s*[\*\-•#]+\s*", "", t)
    t = t.replace("**","").strip()
    return t

def extract_robust_parentheticals(text: str) -> Tuple[List[str], str]:
    parentheticals = []; cleaned_text_parts = []
    nesting_level = 0; last_extraction_end = 0; start_index = 0
    text_to_process = text.strip()

    for i, char in enumerate(text_to_process):
        if char == '(':
            if nesting_level == 0:
                cleaned_text_parts.append(text_to_process[last_extraction_end:i])
                start_index = i
            nesting_level += 1
        elif char == ')':
            if nesting_level > 0:
                nesting_level -= 1
                if nesting_level == 0:
                    content = text_to_process[start_index+1:i].strip()
                    if content:
                        parentheticals.append(content)
                    last_extraction_end = i + 1

    if last_extraction_end < len(text_to_process):
        cleaned_text_parts.append(text_to_process[last_extraction_end:])

    cleaned_text = "".join(cleaned_text_parts).strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip('., ')
    return parentheticals, cleaned_text


# ===========================================================
# SECTION 2: DYNAMIC PROFILE SYSTEM (Stable)
# ===========================================================

# Define Regex patterns for safer word boundaries (Unicode aware)
SAFE_BOUNDARY_START = r"(?<!\w)" 
SAFE_BOUNDARY_END = r"(?!\w)" 

def initialize_context(profile_name: str) -> Dict[str, Any]:
    # (Logic remains stable)
    profile = profiles.load_profile(profile_name)
    if not profile:
        raise ValueError(f"Không thể tải profile: {profile_name}")
        
    # Compile Regexes
    rules = profile['parsing_rules']
    lexicons = profile['lexicons']
    UPPER_CHARS = rules['upper_chars']; LOWER_CHARS = rules['lower_chars']

    WORD_PART = "[" + UPPER_CHARS + LOWER_CHARS + "'-]+"
    CAP_WORD = "[" + UPPER_CHARS + "]" + WORD_PART + "?"

    SPEAKER_RE = re.compile(
        r"^\s*(?:[\*\-•]+\s*)?(?:\*{1,3})?"
        r"(" + CAP_WORD + r"(?:\s+" + WORD_PART + r"){0,10})"
        r"(?:\s*\([^)]*\))?"
        r"(?:\*{1,3})?\s*:\s*"
        r"(.*)$", re.M
    )

    ALIAS_EXPLICIT_RE = re.compile(
        r"(" + CAP_WORD + r"(?:\s+" + WORD_PART + r"){0,10})\s*\(\s*([" + UPPER_CHARS + r"0-9]{1,12})\s*\)"
    )

    action_label = next((k for k, v in rules['structure_labels'].items() if v == 'action'), 'Hành động')
    arrival_label = next((k for k, v in rules['structure_labels'].items() if v == 'arrival'), 'Sự xuất hiện')

    NARRATIVE_INTRO_RE = re.compile(
        rf"^\s*(?:[\*\-•]*\s*\*?\*?\s*)?(?:{re.escape(action_label)}|{re.escape(arrival_label)})\s*(?:\(.*\))?\s*:\s*(.*)", re.M | re.I
    )

    ALLOWED_LOWERCASE_WORDS = set(word for phrase in lexicons['appearance_phrases'] for word in phrase.lower().split())
    LOWERCASE_GROUP = "(?:" + "|".join(re.escape(w) for w in ALLOWED_LOWERCASE_WORDS) + ")" if ALLOWED_LOWERCASE_WORDS else "a^"

    NARRATIVE_NAME_RE = re.compile(
        r"\b(" + CAP_WORD +
        r"(?:\s+(?:" + CAP_WORD + r"|(?i:" + LOWERCASE_GROUP + r"))){0,5})\b"
    )

    SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

    regexes = {
        "SPEAKER_RE": SPEAKER_RE, "ALIAS_EXPLICIT_RE": ALIAS_EXPLICIT_RE,
        "NARRATIVE_INTRO_RE": NARRATIVE_INTRO_RE, "NARRATIVE_NAME_RE": NARRATIVE_NAME_RE,
        "SENTENCE_SPLIT_RE": SENTENCE_SPLIT_RE
    }
    
    return {
        "profile": profile,
        "rules": rules,
        "lexicons": lexicons,
        "cinematic": profile.get('cinematic_instructions', {}),
        "regexes": regexes
    }

# ===========================================================
# SECTION 3: CORE LOGIC - PARSING & EXTRACTION (V15.1, V15.2 Updates)
# ===========================================================

# ---------- Validation Helpers (Stable) ----------
# (is_invalid_struct_or_blacklist, validate_and_clean_name remain stable)

def is_invalid_struct_or_blacklist(tok:str, ctx: Dict[str, Any])->bool:
    rules = ctx['rules']
    base = tok.strip(": .-"); base = re.sub(r"^\s*[\*\-•#]*\s*", "", base)
    _, clean_base = extract_robust_parentheticals(base)

    if not clean_base: return True
    if clean_base.lower() in [b.lower() for b in rules['character_blacklist']]: return True
    for label in rules['structure_labels'].keys():
        if clean_base.lower().startswith(label.lower()): return True
    return False

def validate_and_clean_name(name: str, ctx: Dict[str, Any], allow_group_agents=False) -> Tuple[bool, str]:
    rules = ctx['rules']; lexicons = ctx['lexicons']
    if not name: return False, ""
    if is_invalid_struct_or_blacklist(name, ctx): return False, name

    is_group_agent = name.lower() in [g.lower() for g in rules.get('group_agents', [])]
    if is_group_agent and not allow_group_agents: return False, name
    if is_group_agent and allow_group_agents: return True, name

    prop_set = set(p.lower() for p in lexicons['props_list'])
    if name.lower() in prop_set: return False, name

    cleaned_name = name
    if not is_group_agent:
        for suffix in rules['invalid_name_suffixes']:
            if cleaned_name.lower().endswith(" " + suffix.lower()):
                cleaned_name = cleaned_name[:-(len(suffix)+1)].strip(); break
        for prefix in rules['invalid_name_prefixes']:
            if cleaned_name.lower().startswith(prefix.lower() + " "):
                cleaned_name = cleaned_name[len(prefix)+1:].strip(); break

    if not cleaned_name or is_invalid_struct_or_blacklist(cleaned_name, ctx): return False, name

    is_group_agent_cleaned = cleaned_name.lower() in [g.lower() for g in rules.get('group_agents', [])]
    if is_group_agent_cleaned and not allow_group_agents: return False, cleaned_name
    if cleaned_name.lower() in prop_set: return False, name

    return True, cleaned_name


# --- Phrase Mining Helpers (V15.1 Updates) ---

def mine_phrases(text: str, phrase_list: List[str]) -> List[str]:
    # (Logic remains stable from V13.3)
    found = set()
    sorted_phrases = sorted(phrase_list, key=len, reverse=True)
    canonical_text = canonicalize(text)

    for phrase in sorted_phrases:
        canonical_phrase = canonicalize(phrase)
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(canonical_phrase)}{SAFE_BOUNDARY_END}"
        
        if re.search(pattern, canonical_text, flags=re.I):
            is_subsumed = False
            for existing in found:
                existing_pattern = rf"{SAFE_BOUNDARY_START}{re.escape(canonical_phrase)}{SAFE_BOUNDARY_END}"
                if re.search(existing_pattern, canonicalize(existing), flags=re.I):
                    is_subsumed = True; break
            if not is_subsumed:
                to_remove = set()
                for existing in found:
                    reverse_pattern = rf"{SAFE_BOUNDARY_START}{re.escape(canonicalize(existing))}{SAFE_BOUNDARY_END}"
                    if re.search(reverse_pattern, canonical_phrase, flags=re.I):
                        to_remove.add(existing)
                found -= to_remove
                found.add(phrase)
    return sorted(list(found))

def mine_appearance(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons']['appearance_phrases'])

# V15.1: Specific miners for Stance/Modifier
def mine_dynamic_actions(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons'].get('dynamic_actions', []))

def mine_stances(text: str, ctx: Dict[str, Any]) -> List[str]:
     return mine_phrases(text, ctx['lexicons'].get('stances', []))

def mine_posture_modifiers(text: str, ctx: Dict[str, Any]) -> List[str]:
     return mine_phrases(text, ctx['lexicons'].get('posture_modifiers', []))

# Helper for combined mining (used in classification)
def mine_actions(text: str, ctx: Dict[str, Any]) -> List[str]:
     return mine_phrases(text, ctx['lexicons']['action_phrases'])

def mine_emotions(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons']['emotion_phrases'])

def mine_cinematic_instructions(text: str, ctx: Dict[str, Any]) -> Dict[str, List[str]]:
    # (Logic stable)
    if not text: return {}
    instructions = {"camera": [], "vfx_sfx": [], "meta": []}
    config = ctx['cinematic']
    
    for keyword, tag in config.get('camera_moves', {}).items():
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(keyword)}{SAFE_BOUNDARY_END}"
        if re.search(pattern, text, re.I): instructions["camera"].append(tag)
            
    for keyword, tag in config.get('vfx_sfx', {}).items():
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(keyword)}{SAFE_BOUNDARY_END}"
        if re.search(pattern, text, re.I): instructions["vfx_sfx"].append(tag)
            
    for pattern, tag in config.get('meta_types', {}).items():
        if re.search(pattern, text, re.I): instructions["meta"].append(tag)
            
    return {k: sorted(list(set(v))) for k, v in instructions.items() if v}


# ---------- Scene Detection (Stable) ----------
def detect_scenes(text:str, ctx: Dict[str, Any])->List[Dict[str,Any]]:
    # (Logic remains stable from V13)
    SCENE_HEADER_PATS = ctx['rules']['scene_header_patterns']
    canonical_text = canonicalize(text); lines = canonical_text.splitlines(); idxs=[]
    for i,L in enumerate(lines):
        if any(re.search(p, L.strip(), flags=re.I) for p in SCENE_HEADER_PATS): idxs.append(i)

    if not idxs: return [{"Scene_ID":1,"Title":"Untitled","Raw":canonical_text}]

    idxs.append(len(lines)); out=[]
    
    if idxs[0] > 0:
        body = "\n".join(lines[0:idxs[0]]).strip()
        if body:
             first_title = clean_title(lines[idxs[0]].strip())
             title = first_title if "MỞ ĐẦU" in first_title.upper() else "CẢNH MỞ ĐẦU (Suy luận)"
             out.append({"Scene_ID": 1, "Title": title, "Raw": body})

    start_scene_id = len(out) + 1
    
    for si,(a,b) in enumerate(zip(idxs, idxs[1:]), start=start_scene_id):
        title = clean_title(lines[a].strip())
        body  = "\n".join(lines[a+1:b]).strip()
        
        if out and title == out[-1]["Title"]:
             if body:
                out[-1]["Raw"] += "\n" + body
             continue

        if body: 
            out.append({"Scene_ID":si,"Title":title,"Raw":body})
            
    return out

# ---------- Global Pass: Character Consolidation (V15.1, V15.2 Updates) ----------

def global_character_pass(full_text:str, ctx: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, Dict], Set[str], Dict[str, str]]:
    """(V15) Consolidates characters, performs role analysis, and identifies ambiguous references."""
    # (Extraction logic 1-3 stable)
    text = canonicalize(full_text)
    raw_terms = set(); explicit_aliases = {}
    regexes = ctx['regexes']; rules = ctx['rules']
    CONTEXTUAL_REFS = rules.get('contextual_references', {})

    def process_term(name, alias=None):
        is_valid, cleaned_name = validate_and_clean_name(name, ctx, allow_group_agents=False)
        if is_valid:
            raw_terms.add(cleaned_name)
            if alias:
                is_valid_a, cleaned_alias = validate_and_clean_name(alias, ctx, allow_group_agents=False)
                if is_valid_a:
                     raw_terms.add(cleaned_alias); explicit_aliases[cleaned_alias] = cleaned_name

    # 1. Extraction (Stable)
    for m in regexes['ALIAS_EXPLICIT_RE'].finditer(text):
        name = m.group(1).strip(); alias = m.group(2).strip()
        if len(alias.split()) == 1: process_term(name, alias)

    for m in regexes['SPEAKER_RE'].finditer(text):
        name = m.group(1).strip()
        _, clean_name = extract_robust_parentheticals(name)
        process_term(clean_name)

    for m in regexes['NARRATIVE_INTRO_RE'].finditer(text):
        intro_text = m.group(1)
        for n_match in regexes['NARRATIVE_NAME_RE'].finditer(intro_text):
            potential_name = n_match.group(1).strip()
            is_valid, cleaned_name = validate_and_clean_name(potential_name, ctx, allow_group_agents=False)
            if is_valid:
               is_known = any(t.lower() == cleaned_name.lower() for t in raw_terms)
               if len(cleaned_name.split()) >= 2 or is_known:
                   raw_terms.add(cleaned_name)

    # 2. Resolution & 3. Consolidation (Stable)
    resolved_names = set()
    for term in raw_terms:
        resolved = term
        for alias, name in explicit_aliases.items():
            if alias.lower() == term.lower(): resolved = name; break
        resolved_names.add(resolved)

    sorted_names = sorted(list(resolved_names), key=len)
    canonical_map = {name: name for name in resolved_names}

    for i, name_a in enumerate(sorted_names):
        for name_b in sorted_names[i+1:]:
            if re.search(rf"\b{re.escape(name_a.lower())}\b", name_b.lower()):
                 canonical_map[name_a] = canonical_map[name_b]

    # V15.2: Role Determination (Internal Roles for Resolution)
    character_roles_internal = {}
    canonical_names = set(canonical_map.values())
    for name in canonical_names:
        role = "student"
        if any(t in name.lower() for t in rules['role_self_teacher']): 
            role = "teacher"
        # Heuristic based on script context (Crucial for Anaphora)
        elif "thỏ" in name.lower() or "vịt" in name.lower(): 
             role = "female_child"
        elif "gấu" in name.lower():
             role = "male_child"
        character_roles_internal[name] = role

    # V15.2: Contextual Reference Consolidation (Global vs Local)
    global_anaphora_map = {}
    local_anaphora_terms = set()

    for ref_term, ref_role in CONTEXTUAL_REFS.items():
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(ref_term)}{SAFE_BOUNDARY_END}"
        if re.search(pattern, text, flags=re.I):
            potential_matches = [name for name in canonical_names if character_roles_internal.get(name) == ref_role]
            
            if len(potential_matches) == 1:
                # Unambiguous -> Map Globally
                global_anaphora_map[ref_term.lower()] = potential_matches[0]
            elif len(potential_matches) > 1:
                # Ambiguous -> Flag for Local Resolution
                local_anaphora_terms.add(ref_term.lower())

    # 4. Final Mapping
    global_term_to_canonical_map = {}
    for term in raw_terms:
        resolved = term
        for alias, name in explicit_aliases.items():
            if alias.lower() == term.lower(): resolved = name; break
        
        canonical = None
        for variation, canon in canonical_map.items():
            if variation.lower() == resolved.lower(): canonical = canon; break

        if canonical: global_term_to_canonical_map[term] = canonical

    # Add Global Anaphora maps
    for term, canon in global_anaphora_map.items():
         if term not in global_term_to_canonical_map:
              if canon in canonical_names:
                global_term_to_canonical_map[term] = canon

    # 5. Building Global Character Registry (V15.1 Update)
    global_characters = {}
    
    for c_name in canonical_names:
        internal_role = character_roles_internal.get(c_name, "student")
        
        # Standardize role for output (Map internal roles back to standardized ones if needed)
        if internal_role in ["female_child", "male_child"]:
             standardized_role = "student"
        else:
             standardized_role = internal_role

        
        aliases = []
        for term, canon in global_term_to_canonical_map.items():
            if canon == c_name and term.lower() != c_name.lower(): 
                aliases.append(term)

        inherent_appearance = mine_appearance(c_name, ctx)

        # V15.1: Update structure for Stance/Modifier
        global_characters[c_name] = {
            "role": standardized_role,
            "aliases": sorted(aliases),
            "slug": slugify(c_name),
            "inherent_attributes": {
                "Appearance": inherent_appearance,
                "Actions": [],
                "Emotions": [],
                "Stance": [],
                "Posture_Modifiers": []
            }
        }

    # 6. Building Global Group Agent Registry (V15.1 Update)
    global_group_agents = {}
    for agent_name in rules.get('group_agents', []):
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(agent_name)}{SAFE_BOUNDARY_END}"
        if re.search(pattern, text, flags=re.I):
             inherent_appearance = mine_appearance(agent_name, ctx)
             # V15.1: Update structure
             global_group_agents[agent_name] = {
                "slug": slugify(agent_name),
                "type": "group",
                "inherent_attributes": { 
                    "Appearance": inherent_appearance,
                    "Actions": [],
                    "Emotions": [],
                    "Stance": [],
                    "Posture_Modifiers": []
                }
            }
    
    # V15.2: Create Role Map (Slug -> Internal Role) for Local Anaphora Resolution
    role_map = {}
    for name, data in global_characters.items():
        # Use internal roles here for resolution logic
        role_map[data['slug']] = character_roles_internal.get(name, "student")

    # V15.2: Return the necessary components
    return global_term_to_canonical_map, global_characters, global_group_agents, local_anaphora_terms, role_map

# ---------- Context Derivation (Stable) ----------
# (derive_context remains stable)
def derive_context(scene_text:str, ctx: Dict[str, Any])->Dict[str,Any]:
    s = scene_text; lexicons = ctx['lexicons']; rules = ctx['rules']

    def pick(label_regex:str)->str:
        pat = rf"^\s*(?:\*{{0,3}})?\s*(?:{label_regex})\s*(?:\*{{0,3}})?\s*:\s*(?P<val>.+)$"
        m = re.search(pat, s, flags=re.I|re.M)
        return m.group("val").strip() if m else ""

    setting_label = next((k for k, v in rules['structure_labels'].items() if v == 'setting'), 'Bối cảnh')
    setting_line = pick(re.escape(setting_label) + r"|Setting")

    if setting_line:
        setting = setting_line
    else:
        setting = "không gian lớp học ngoài trời (suy luận)" if ctx['profile']['language'] == 'vi' else "undefined location (inferred)"

    # Prop consolidation logic
    low = s.lower(); raw_props = set()
    sorted_props_list = sorted(lexicons['props_list'], key=len, reverse=True)

    for kw in sorted_props_list:
        canonical_kw = canonicalize(kw)
        pattern = rf"{SAFE_BOUNDARY_START}{re.escape(canonical_kw)}{SAFE_BOUNDARY_END}"
        if re.search(pattern, low, re.I):
            raw_props.add(kw)

    sorted_raw_props = sorted(list(raw_props), key=len); final_props = set(raw_props)
    for i, prop_a in enumerate(sorted_raw_props):
        for prop_b in sorted_raw_props[i+1:]:
            if prop_a in prop_b:
                 if prop_a in final_props:
                     try: final_props.remove(prop_a)
                     except KeyError: pass
                 break

    # TOD and Tone logic
    tod = "morning"
    if ctx['profile']['language'] == 'vi' and ("sáng sớm" in low or "nắng sớm" in low): 
        tod = "early_morning"

    tone=[];
    for key, val in lexicons['tone_map']:
        if key in low: tone.append(val)

    if not tone: tone=["warm","gentle"]

    return {"setting":setting,"props":sorted(list(final_props)),"time_of_day":tod,"tone":sorted(list(set(tone))), "setting_line": setting_line}


# ===========================================================
# SECTION 4: BEAT & SHOT PROCESSING (V15.1, V15.2, V15.3, V15.5 Updates)
# ===========================================================

LINE_TYPES = {
    "DIALOGUE": "dialogue", "ACTION": "action", "STRUCTURE": "structure",
    "META": "meta", "IGNORE": "ignore"
}

def classify_line(line:str, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
    # (Logic remains stable, utilizing combined mine_actions)
    raw = line.strip()
    if not raw: return LINE_TYPES["IGNORE"], "", ""
    regexes = ctx['regexes']; rules = ctx['rules']

    # 1. Check Dialogue
    if regexes['SPEAKER_RE'].match(raw):
        speaker_part = regexes['SPEAKER_RE'].match(raw).group(1).strip()
        _, clean_speaker = extract_robust_parentheticals(speaker_part)
        is_valid, _ = validate_and_clean_name(clean_speaker, ctx, allow_group_agents=True)
        if is_valid: return LINE_TYPES["DIALOGUE"], raw, ""

    # 2. Check Structure Labels
    clean_for_struct_check = re.sub(r"^\s*[\*\-•]*\s*\*?\*?\s*", "", raw)

    for label, tag in rules['structure_labels'].items():
        if re.match(rf"^{re.escape(label)}\s*(?:\(.*\))?\s*:", clean_for_struct_check, re.I):
            if tag == 'action': return LINE_TYPES["ACTION"], raw, tag
            return LINE_TYPES["STRUCTURE"], raw, tag

    # 3. Check for META vs ACTION conflict
    has_narrative = mine_actions(raw, ctx) or mine_emotions(raw, ctx) or mine_appearance(raw, ctx)
    instructions = mine_cinematic_instructions(raw, ctx)
    
    if instructions.get("meta") and not has_narrative:
         return LINE_TYPES["META"], raw, "meta_instruction"

    # 4. Default to Action.
    return LINE_TYPES["ACTION"], raw, ""


def safe_clean_labeled_content(line:str, ctx: Dict[str, Any]) -> str:
    rules = ctx['rules']
    clean_line = re.sub(r"^\s*(\*{{1,3}})?(.*?)\1?\s*$", r"\2", line).strip()

    # Handle standard labels (e.g., Hành động:)
    for label in rules['structure_labels'].keys():
        pattern = rf"^(?:[\*\-•]\s*)?{re.escape(label)}\s*(?:\(.*\))?\s*:\s*"
        if re.match(pattern, clean_line, re.I):
            cleaned = re.sub(pattern, "", clean_line, count=1, flags=re.I).strip()
            return cleaned
    
    # Handle lines starting with (Parenthetical) Action: (e.g. (Slow-motion) Hành động:)
    if clean_line.startswith('('):
        parentheticals, remaining_text = extract_robust_parentheticals(clean_line)
        if remaining_text.lower().startswith('hành động:'):
             cleaned = remaining_text[len('hành động:'):].strip()
             return cleaned

    return line.strip()

# ---------- Narrative Analysis (V15.1, V15.2, V15.3, V15.5 Updates) ----------

def analyze_narrative_attributes(
    text_lines: List[str], 
    global_context: Dict[str, Any], 
    ctx: Dict[str, Any],
    recent_subject_slug: Optional[str] = None # V15.2
) -> Dict[str, Dict[str, List[str]]]:
    """(V15) Analyzes narrative text: Stance/Modifier, Local Anaphora, Single-Agent Fallback, Prop-Aware Filtering."""
    
    AGENT_SLUG_LOOKUP = global_context['Agent_Slug_Lookup']
    # V15.2: Load Anaphora Context
    LOCAL_ANAPHORA_TERMS = global_context.get('Local_Anaphora_Terms', set())
    ROLE_MAP = global_context.get('Role_Map', {})

    attributes_by_slug = {}
    regexes = ctx['regexes']; rules = ctx['rules']; lexicons = ctx['lexicons']
    NEGATION_KEYWORDS = rules.get('negation_keywords', [])
    CONTEXTUAL_REFS = rules.get('contextual_references', {})

    full_text = " ".join(text_lines)
    sentences = regexes['SENTENCE_SPLIT_RE'].split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences and full_text.strip():
        sentences = [full_text.strip()]

    # V15.2: Prepare terms
    sorted_explicit_terms = sorted(AGENT_SLUG_LOOKUP.keys(), key=len, reverse=True)
    sorted_local_anaphora_terms = sorted(list(LOCAL_ANAPHORA_TERMS), key=len, reverse=True)

    # V15.1/V15.5: Lexicon preparation
    sorted_dynamic_actions = sorted(lexicons.get('dynamic_actions', []), key=len, reverse=True)
    sorted_stances = sorted(lexicons.get('stances', []), key=len, reverse=True)
    sorted_modifiers = sorted(lexicons.get('posture_modifiers', []), key=len, reverse=True)
    sorted_emotions = sorted(lexicons['emotion_phrases'], key=len, reverse=True)
    sorted_appearance = sorted(lexicons['appearance_phrases'], key=len, reverse=True)
    sorted_props = sorted(lexicons['props_list'], key=len, reverse=True) # V15.5

    def initialize_slug(slug):
        if slug not in attributes_by_slug:
            # V15.1: Structure
            attributes_by_slug[slug] = {"Actions": [], "Stance": [], "Posture_Modifiers": [], "Emotions": [], "Appearance": []}

    # V15.2: Enhanced find_entities
    def find_entities(sentence, terms, type_label):
        entities = []; canonical_sentence = canonicalize(sentence)
        for term in terms:
            canonical_term = canonicalize(term)
            pattern = rf"{SAFE_BOUNDARY_START}{re.escape(canonical_term)}{SAFE_BOUNDARY_END}"
            
            for match in re.finditer(pattern, canonical_sentence, re.I):
                entity_value = term
                
                # Determine the actual value based on the type
                if type_label == "AGENT_EXPLICIT":
                     entity_value = AGENT_SLUG_LOOKUP.get(canonical_term.lower())
                elif type_label == "AGENT_LOCAL_ANAPHORA":
                     # V15.2: Local Resolution Logic
                     ref_term = canonical_term.lower()
                     expected_role = CONTEXTUAL_REFS.get(ref_term)
                     entity_value = None

                     if recent_subject_slug and expected_role:
                        # Check if the recent subject matches the expected role
                        recent_subject_role = ROLE_MAP.get(recent_subject_slug)
                        if recent_subject_role == expected_role:
                            entity_value = recent_subject_slug
                
                if entity_value:
                    is_overlapped = False
                    # (Overlap check logic remains stable)
                    existing_to_remove = []
                    for existing in entities:
                        is_agent_type = type_label.startswith("AGENT")
                        is_existing_agent_type = existing['type'].startswith("AGENT")

                        # Only check overlap if types are compatible (e.g. Agents vs Agents)
                        if is_agent_type == is_existing_agent_type:
                            if match.start() >= existing['start'] and match.end() <= existing['end']:
                                is_overlapped = True; break
                            if match.start() <= existing['start'] and match.end() >= existing['end']:
                                existing_to_remove.append(existing)

                    if existing_to_remove:
                         for item in existing_to_remove:
                              entities.remove(item)

                    if not is_overlapped:
                        # If it's a resolved Anaphora, standardize type to "AGENT"
                        final_type = "AGENT" if type_label.startswith("AGENT") else type_label
                        entities.append({"start": match.start(), "end": match.end(), "value": entity_value, "type": final_type})
        return entities

    for sentence in sentences:
        # V15.2: Find all types of entities
        explicit_agents = find_entities(sentence, sorted_explicit_terms, "AGENT_EXPLICIT")
        resolved_local_agents = find_entities(sentence, sorted_local_anaphora_terms, "AGENT_LOCAL_ANAPHORA")
        
        agents = explicit_agents + resolved_local_agents

        # V15.1/V15.5: Find attributes and props
        dynamic_actions = find_entities(sentence, sorted_dynamic_actions, "ACTION")
        stances = find_entities(sentence, sorted_stances, "STANCE")
        modifiers = find_entities(sentence, sorted_modifiers, "MODIFIER")
        emotions = find_entities(sentence, sorted_emotions, "EMOTION")
        appearances = find_entities(sentence, sorted_appearance, "APPEARANCE")
        props = find_entities(sentence, sorted_props, "PROP") # V15.5

        # V15.5: Prop-Aware Attribute Filtering
        filtered_appearances = []
        for app_entity in appearances:
            is_part_of_prop = False
            for prop_entity in props:
                proximity_threshold = 5 # Allow short connecting words
                
                # Case 1: Appearance precedes Prop (e.g., "xanh" [Súp lơ])
                if app_entity['end'] <= prop_entity['start'] and (prop_entity['start'] - app_entity['end']) < proximity_threshold:
                     is_part_of_prop = True; break
                # Case 2: Prop precedes Appearance (e.g., [Súp lơ] "xanh")
                if prop_entity['end'] <= app_entity['start'] and (app_entity['start'] - prop_entity['end']) < proximity_threshold:
                     is_part_of_prop = True; break
            
            if not is_part_of_prop:
                filtered_appearances.append(app_entity)


        all_entities = sorted(agents + dynamic_actions + stances + modifiers + emotions + filtered_appearances, key=lambda x: x['start'])

        # V15.1: Define Attribute Type Map
        attr_type_map = {"ACTION": "Actions", "STANCE": "Stance", "MODIFIER": "Posture_Modifiers", "EMOTION": "Emotions", "APPEARANCE": "Appearance"}

        # V15.3: Single-Agent Association Fallback
        if len(agents) == 1:
            agent_entity = agents[0]
            slug = agent_entity['value']
            initialize_slug(slug)
            
            # Associate ALL non-agent entities found in the sentence
            for entity in all_entities:
                if entity['type'] == "AGENT": continue

                # Check for negation (simple proximity check)
                is_negated = False
                pre_segment = sentence[max(0, entity['start']-15):entity['start']]
                
                for kw in NEGATION_KEYWORDS:
                    pattern = rf"{SAFE_BOUNDARY_START}{re.escape(kw)}{SAFE_BOUNDARY_END}"
                    if re.search(pattern, pre_segment, re.I):
                        is_negated = True; break
                
                if not is_negated:
                    attr_key = attr_type_map.get(entity['type'])
                    if attr_key and entity['value'] not in attributes_by_slug[slug][attr_key]:
                        attributes_by_slug[slug][attr_key].append(entity['value'])
            
            # Skip Bidirectional search if fallback was used
            continue

        # Bidirectional Proximity Association (Used when >1 or 0 agents)
        for i, entity in enumerate(all_entities):
            if entity['type'] != "AGENT":
                continue

            slug = entity['value']
            initialize_slug(slug)

            # Search Forward (Right)
            for j in range(i + 1, len(all_entities)):
                next_entity = all_entities[j]
                if next_entity['type'] == "AGENT": break 

                segment = sentence[entity['end']:next_entity['start']]
                is_negated = False
                for kw in NEGATION_KEYWORDS:
                    pattern = rf"{SAFE_BOUNDARY_START}{re.escape(kw)}{SAFE_BOUNDARY_END}"
                    if re.search(pattern, segment, re.I):
                        is_negated = True; break

                if not is_negated:
                    attr_key = attr_type_map.get(next_entity['type'])
                    if attr_key and next_entity['value'] not in attributes_by_slug[slug][attr_key]:
                        attributes_by_slug[slug][attr_key].append(next_entity['value'])

            # Search Backward (Left)
            for j in range(i - 1, -1, -1):
                prev_entity = all_entities[j]
                if prev_entity['type'] == "AGENT": break

                segment = sentence[prev_entity['end']:entity['start']]
                is_negated = False
                for kw in NEGATION_KEYWORDS:
                    pattern = rf"{SAFE_BOUNDARY_START}{re.escape(kw)}{SAFE_BOUNDARY_END}"
                    if re.search(pattern, segment, re.I):
                        is_negated = True; break

                if not is_negated:
                    # V15.1: Allow Stance/Modifier/Emotion/Action preceding the agent
                    attr_key = attr_type_map.get(prev_entity['type'])
                    if attr_key in ["Emotions", "Stance", "Posture_Modifiers", "Actions"]:
                         if prev_entity['value'] not in attributes_by_slug[slug][attr_key]:
                            attributes_by_slug[slug][attr_key].append(prev_entity['value'])

    
    # Sort results
    for slug in attributes_by_slug:
        for key in attributes_by_slug[slug]:
            attributes_by_slug[slug][key] = sorted(attributes_by_slug[slug][key])

    return attributes_by_slug


# ---------- Beat Extraction (V15.1 Updates) ----------

# V15.2: Updated signature to accept Global Context
def extract_beats(scene_text:str, global_context: Dict[str, Any], ctx: Dict[str, Any]) -> List[Dict[str,Any]]:
    """(V15) Segments the scene into beats. Defers detailed Action analysis to Shot processing."""
    
    AGENT_SLUG_LOOKUP = global_context['Agent_Slug_Lookup']

    # (Classification logic remains stable)
    lines = scene_text.splitlines()
    classified_lines = []
    for line in lines:
        l_type, content, tag = classify_line(line, ctx)
        if l_type != LINE_TYPES["IGNORE"]:
            classified_lines.append({"type": l_type, "raw": content, "tag": tag})

    if not classified_lines:
        return [] 

    beats = []; beat_id = 1; regexes = ctx['regexes']
    VO_PATTERNS = [r"\(V\.O\)", r"\(V\.O\.\)", r"\(Lồng tiếng\)"]

    for group_type, group_iter in itertools.groupby(classified_lines, key=lambda x: x['type']):
        group = list(group_iter)

        if group_type == LINE_TYPES["STRUCTURE"]:
            for item in group:
                tag = item['tag']
                if tag == 'setting': continue
                cleaned_text = safe_clean_labeled_content(item['raw'], ctx)
                if cleaned_text:
                    beat = {"id":f"B{beat_id}","type":tag,"text_lines":[cleaned_text]}
                    # V15: Analysis is deferred to process_scene for all beat types to leverage context.
                    beats.append(beat); beat_id += 1

        elif group_type == LINE_TYPES["DIALOGUE"]:
            # (V15.1 Updates for attributes)
            text_lines = [item['raw'] for item in group]
            dialogue_lines = []

            for line in text_lines:
                match = regexes['SPEAKER_RE'].match(line)
                if match:
                    speaker_raw = match.group(1).strip()
                    dialogue_content = match.group(2).strip()

                    # Speaker resolution
                    _, clean_speaker_name = extract_robust_parentheticals(speaker_raw)
                    canonical_speaker = canonicalize(clean_speaker_name).lower()
                    speaker_slug = AGENT_SLUG_LOOKUP.get(canonical_speaker, slugify(clean_speaker_name))

                    # Extraction and cleaning
                    parentheticals_raw, clean_line = extract_robust_parentheticals(dialogue_content)
                    
                    # Handle quotes
                    clean_line = clean_line.strip()
                    if (clean_line.startswith('"') and clean_line.endswith('"')) or \
                       (clean_line.startswith("'") and clean_line.endswith("'")):
                        if len(clean_line) > 1:
                           clean_line = clean_line[1:-1].strip()
                    
                    parenthetical_text = "; ".join(parentheticals_raw) if parentheticals_raw else None

                    # Analysis (V15.1 Update)
                    instructions = {}
                    # V15.1: Initialize structure
                    attributes = {"Actions": [], "Stance": [], "Posture_Modifiers": [], "Emotions": [], "Appearance": []}
                    
                    # Process EACH parenthetical
                    for p_text in parentheticals_raw:
                        # V15.1: Use specific miners
                        attributes["Actions"].extend(mine_dynamic_actions(p_text, ctx))
                        attributes["Stance"].extend(mine_stances(p_text, ctx))
                        attributes["Posture_Modifiers"].extend(mine_posture_modifiers(p_text, ctx))
                        attributes["Emotions"].extend(mine_emotions(p_text, ctx))
                        attributes["Appearance"].extend(mine_appearance(p_text, ctx))
                        
                        p_instr = mine_cinematic_instructions(p_text, ctx)
                        for k, v in p_instr.items():
                            instructions.setdefault(k, []).extend(v)

                    # V.O. Detection
                    is_vo = False
                    if any(re.search(p, speaker_raw, re.I) for p in VO_PATTERNS):
                         is_vo = True

                    dialogue_entry = {
                        "Speaker_Slug": speaker_slug,
                        "Parenthetical": parenthetical_text,
                        "Line": clean_line,
                        "Is_Voice_Over": is_vo
                    }
                    
                    # Final cleanup
                    final_attrs = {k: sorted(list(set(v))) for k, v in attributes.items() if v}
                    if final_attrs:
                        dialogue_entry["attributes"] = final_attrs
                    
                    final_instr = {k: sorted(list(set(v))) for k, v in instructions.items() if v}
                    if final_instr:
                         dialogue_entry["instructions"] = final_instr

                    dialogue_lines.append(dialogue_entry)

            beats.append({"id":f"B{beat_id}","type":"dialogue","text_lines":text_lines, "dialogue_lines": dialogue_lines})
            beat_id += 1

        elif group_type == LINE_TYPES["ACTION"] or group_type == LINE_TYPES["META"]:
            text_lines = [item['raw'] for item in group]
            beat_type = "establish" if (beat_id == 1 and not beats and group_type == LINE_TYPES["ACTION"]) else group[0]['tag'] or group_type.lower()

            beat = {"id":f"B{beat_id}","type":beat_type,"text_lines":text_lines}

            # V15: Analysis is DEFERRED to process_scene (Shot level) for ACTION beats
            
            # Only mine instructions here if needed
            instructions = mine_cinematic_instructions(" ".join(text_lines), ctx)
            if instructions: beat["instructions"] = instructions

            beats.append(beat); beat_id += 1

    # Post-processing
    if beats and beats[0]['type'] == 'action':
         beats[0]['type'] = 'establish'

    return beats

# ===========================================================
# SECTION 5: SCENE PROCESSING & CINEMATIC LOGIC (V15.0, V15.1, V15.4 Updates)
# ===========================================================

# (segment_shots, determine_shot_type remain stable)
def segment_shots(beats: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Segments beats into shots, splitting dialogue by speaker turn."""
    shots = []; shot_id = 1; regexes = ctx['regexes']

    for beat in beats:
        # V15.4: Handle dialogue (implicit dialogue is reclassified to 'dialogue' during process_scene)
        if beat['type'] == 'dialogue':
            dialogue_lines = beat.get('dialogue_lines', [])
            
            # Heuristic: Split if multiple lines exist.
            should_split = len(dialogue_lines) > 1

            if not should_split:
                 shots.append({"Shot_ID": f"S{shot_id}", "Beats": [beat]}); shot_id += 1
            else:
                 # Split explicit dialogue
                 for i, line in enumerate(dialogue_lines):
                    sub_beat = {
                        'id': f"{beat['id']}.{i+1}",
                        'type': 'dialogue',
                        'dialogue_lines': [line]
                    }
                    # (Text line mapping logic stable)
                    if i < len(beat['text_lines']):
                        sub_beat['text_lines'] = [beat['text_lines'][i]]
                    else:
                        sub_beat['text_lines'] = [f"[Dialogue Turn] {line['Speaker_Slug']}: {line['Line']}"]

                    shots.append({"Shot_ID": f"S{shot_id}", "Beats": [sub_beat]}); shot_id += 1

        elif beat['type'] in ['meta', 'setting']:
            shots.append({"Shot_ID": f"S{shot_id}", "Beats": [beat]}); shot_id += 1
        else:
            # Narrative beat splitting
            text = " ".join(beat['text_lines'])
            sentences = regexes['SENTENCE_SPLIT_RE'].split(text)
            
            # Robust cleaning of split sentences
            cleaned_sentences = []
            for s in sentences:
                s = s.strip()
                # Remove stray leading/trailing artifacts that might occur during splitting
                s = re.sub(r"^[(\[\{•\*\- ]+", "", s)
                # s = re.sub(r"[)\]\}\.\, ]+$", "", s) # Careful not to remove intentional punctuation
                if s:
                    cleaned_sentences.append(s)

            if not cleaned_sentences and text.strip():
                 cleaned_sentences = [text.strip()]

            if len(cleaned_sentences) <= 1:
                shots.append({"Shot_ID": f"S{shot_id}", "Beats": [beat]}); shot_id += 1
            else:
                # Create sub-beats
                for i, sentence in enumerate(cleaned_sentences):
                    sub_beat = copy.deepcopy(beat) 
                    sub_beat['id'] = f"{beat['id']}.{i+1}"
                    sub_beat['text_lines'] = [sentence]
                    # Clear attributes for re-analysis in process_scene
                    if 'attributes' in sub_beat: del sub_beat['attributes']
                    if 'instructions' in sub_beat: del sub_beat['instructions']
                    shots.append({"Shot_ID": f"S{shot_id}", "Beats": [sub_beat]}); shot_id += 1
    return shots

def determine_shot_type(shot_composition: Dict[str, Any], instructions: Dict[str, List[str]]) -> str:
    if instructions and instructions.get("camera"):
        camera_instr = instructions["camera"]
        if "close_up_shot" in camera_instr: return "CU/ECU"

    active_chars = sum(1 for d in shot_composition.get("Characters", {}).values() if d.get("Status") == 'active')
    active_groups = sum(1 for d in shot_composition.get("Groups", {}).values() if d.get("Status") == 'active')
    total_active = active_chars + active_groups

    if total_active == 0: return "WS"
    if total_active == 1: return "MCU"
    if total_active == 2: return "MS"
    return "WS"

# ---------- Persistence Helpers (V15.1 Updates) ----------

def resolve_emotion_conflicts(current_emotions: List[str], new_emotions: List[str], conflict_groups: List[Dict[str, Any]]) -> List[str]:
    # (Stable)
    if not new_emotions or not conflict_groups:
        return sorted(list(set(current_emotions + new_emotions)))
    resolved = list(current_emotions)
    for new_emotion in new_emotions:
        conflicts_to_remove = set()
        for group in conflict_groups:
            if new_emotion in group['emotions']:
                for existing_emotion in resolved:
                    if existing_emotion in group['emotions'] and existing_emotion != new_emotion:
                        conflicts_to_remove.add(existing_emotion)
        resolved = [e for e in resolved if e not in conflicts_to_remove]
        if new_emotion not in resolved:
            resolved.append(new_emotion)
    return sorted(resolved)

# V15.1: Helper for Stance Conflict Resolution
def resolve_stance_conflicts(current_stance: Optional[str], new_stance: Optional[str], dynamic_actions: List[str], conflict_map: Dict[str, List[str]]) -> Optional[str]:
    """(V15.1) Resolves stance conflicts (Primary Posture)."""
    if new_stance: return new_stance
    if current_stance and current_stance in conflict_map:
        conflicting_actions = conflict_map[current_stance]
        for action in dynamic_actions:
            if action in conflicting_actions:
                return None
    return current_stance

# V15.1: Helper for Modifier Conflict Resolution
def resolve_modifier_conflicts(current_modifiers: List[str], new_modifiers: List[str], dynamic_actions: List[str], conflict_map: Dict[str, List[str]]) -> List[str]:
    """(V15.1) Resolves posture modifier conflicts."""
    resolved = list(current_modifiers)
    for mod in new_modifiers:
        if mod not in resolved:
            resolved.append(mod)
            
    modifiers_to_remove = set()
    for mod in resolved:
        if mod in conflict_map:
            conflicting_actions = conflict_map[mod]
            for action in dynamic_actions:
                if action in conflicting_actions:
                    modifiers_to_remove.add(mod)
                    break
    
    resolved = [mod for mod in resolved if mod not in modifiers_to_remove]
    return sorted(resolved)


def filter_inherent_attributes(attributes: Dict[str, List[str]], inherent_attributes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # (Stable)
    filtered = copy.deepcopy(attributes)
    inherent_appearance_set = set(canonicalize(a).lower() for a in inherent_attributes.get("Appearance", []))
    if "Appearance" in filtered:
        filtered_appearance = [
            attr for attr in filtered["Appearance"] 
            if canonicalize(attr).lower() not in inherent_appearance_set
        ]
        if filtered_appearance:
            filtered["Appearance"] = sorted(filtered_appearance)
        else:
            del filtered["Appearance"]
    return filtered

# ---------- Scene Processing (V15 Main Logic) ----------

# V15.0: Updated signature to accept persistence_tracker
def process_scene(scene_data: Dict[str, Any], global_context: Dict[str, Any], ctx: Dict[str, Any], persistence_tracker: Dict[str, Dict]) -> Dict[str, Any]:
    """
    (V15) Main scene processing pipeline.
    """
    G_CHARS_BY_SLUG = global_context['Global_Characters_By_Slug']
    G_GROUPS_BY_SLUG = global_context['Global_Group_Agents_By_Slug']
    rules = ctx['rules']

    # 1. Context Derivation
    context = derive_context(scene_data['Raw'], ctx)

    # 2. Beat Extraction (V15.2 Applied)
    # V15: Analysis is deferred from here.
    beats = extract_beats(scene_data['Raw'], global_context, ctx)

    # 3. Shot Segmentation
    shots = segment_shots(beats, ctx)

    # 4. Shot Composition & Attribute Persistence Management
    
    # V15.0: persistence_tracker is passed in. We work on it directly.
    
    # Track physical presence in THIS scene, initialized with persisted characters
    scene_presence = set(persistence_tracker.keys())

    # V15.2: Track the most recent active subject (localized to the scene flow)
    recent_active_subject = None 

    # Initial State Injection (V15.1 Update)
    if context.get("setting_line"):
        # V15.2: Analyze initial state. recent_subject is None at start.
        initial_attrs = analyze_narrative_attributes([context["setting_line"]], global_context, ctx, None)
        
        for slug, attrs in initial_attrs.items():
            scene_presence.add(slug)
            if slug not in persistence_tracker:
                # V15.1 Initialize Structure
                persistence_tracker[slug] = {"Appearance": [], "Emotions": [], "Stance": None, "Posture_Modifiers": []}
            
            # Apply states with conflict resolution (Crucial for V15.0/V15.1)
            
            # Appearance
            if 'Appearance' in attrs: 
                persistence_tracker[slug]['Appearance'].extend(attrs.get('Appearance', []))
                persistence_tracker[slug]['Appearance'] = sorted(list(set(persistence_tracker[slug]['Appearance'])))
            
            # Emotions
            if 'Emotions' in attrs: 
                persistence_tracker[slug]['Emotions'] = resolve_emotion_conflicts(persistence_tracker[slug]['Emotions'], attrs.get('Emotions', []), rules.get("emotion_conflict_groups", []))
            
            # V15.1: Stance and Modifiers
            current_dynamic_actions = attrs.get('Actions', []) # Actions in setting might affect posture

            # Stance
            current_stance = persistence_tracker[slug]['Stance']
            new_stance = attrs.get('Stance', [None])[0] if attrs.get('Stance') else None
            resolved_stance = resolve_stance_conflicts(
                current_stance, new_stance, current_dynamic_actions, rules.get("stance_conflict_map", {})
            )
            persistence_tracker[slug]['Stance'] = resolved_stance

            # Modifiers
            current_modifiers = list(persistence_tracker[slug]['Posture_Modifiers'])
            new_modifiers = attrs.get('Posture_Modifiers', [])
            resolved_modifiers = resolve_modifier_conflicts(
                current_modifiers, new_modifiers, current_dynamic_actions, rules.get("modifier_conflict_map", {})
            )
            persistence_tracker[slug]['Posture_Modifiers'] = resolved_modifiers


    # --- Main Shot Processing Loop ---
    for shot in shots:
        shot_composition = {"Characters": {}, "Groups": {}}
        shot_instructions = {}
        active_agents = set()
        shot_attributes = {}
        current_shot_subjects = set()

        # V15.4: Prepare list for potentially modified beats (due to Implicit Dialogue)
        processed_beats = []

        # --- 4.1: Process Beats within the Shot (Analysis, Inference, Re-classification) ---
        for beat in shot['Beats']:
            
            # A. Analysis/Inference for Narrative Beats (V15.2, V15.3, V15.4, V15.5)
            is_narrative_beat = beat['type'] not in ['dialogue', 'meta', 'setting']
            
            # V15: Analyze narrative beats here (as analysis was deferred from extract_beats)
            if is_narrative_beat:
                # Prepare text for analysis (handle labeled actions)
                # Check if it's an explicitly labeled action line that hasn't been split
                if beat.get('tag') == 'action' and not '.' in beat['id']:
                     text_lines_for_analysis = [safe_clean_labeled_content(line, ctx) for line in beat['text_lines']]
                else:
                     text_lines_for_analysis = beat['text_lines']

                # V15.2: Perform analysis using Local Context (V15.3/V15.5 integrated here)
                beat_attrs = analyze_narrative_attributes(
                    text_lines_for_analysis, global_context, ctx, recent_active_subject
                )
                 
                explicit_subjects = set(beat_attrs.keys())
                text = " ".join(text_lines_for_analysis)

                # V15.4: Check for Implicit Dialogue indicators
                is_implicit_dialogue = False
                clean_text = text.strip()
                # Check if starts with dialogue markers OR contains quotes and is reasonably short
                if clean_text.startswith(('"', "'", "-")):
                      is_implicit_dialogue = True
                elif ('"' in clean_text or "'" in clean_text) and len(clean_text.split()) < 25:
                      is_implicit_dialogue = True

                 
                # Subject Inference Logic (Fallback)
                inferred_subject = None
                if not explicit_subjects and recent_active_subject:
                     
                     # Manual mining (V15.1 Structure)
                     actions = mine_dynamic_actions(text, ctx)
                     stances = mine_stances(text, ctx)
                     modifiers = mine_posture_modifiers(text, ctx)
                     emotions = mine_emotions(text, ctx)
                     appearance = mine_appearance(text, ctx)
                     
                     if actions or stances or modifiers or emotions or appearance or is_implicit_dialogue:
                         inferred_subject = recent_active_subject
                         inferred_attrs = {
                             "Actions": actions, "Stance": stances, "Posture_Modifiers": modifiers,
                             "Emotions": emotions, "Appearance": appearance
                         }
                         
                         if inferred_subject not in beat_attrs:
                              beat_attrs[inferred_subject] = {}
                         
                         # Merge inferred attributes
                         for attr_type, attrs_list in inferred_attrs.items():
                             if attrs_list:
                                 beat_attrs[inferred_subject].setdefault(attr_type, []).extend(attrs_list)

                # Update the beat data
                if beat_attrs: beat['attributes'] = beat_attrs
                beat_instr = mine_cinematic_instructions(" ".join(beat['text_lines']), ctx) # Use original lines for instructions
                if beat_instr: beat['instructions'] = beat_instr

                # V15.4: Re-classification Logic (Implicit Dialogue)
                # Determine the final subject (Inferred or Explicit if singular)
                current_subjects = list(beat_attrs.keys())
                final_subject_slug = inferred_subject or (current_subjects[0] if len(current_subjects) == 1 else None)
                 
                if is_implicit_dialogue and final_subject_slug:
                    # Convert Beat to Dialogue
                    beat['type'] = 'dialogue' # Treat as regular dialogue now
                    
                    # Clean the line
                    dialogue_line_text = clean_text.strip("-*• ")
                    if (dialogue_line_text.startswith('"') and dialogue_line_text.endswith('"')) or \
                       (dialogue_line_text.startswith("'") and dialogue_line_text.endswith("'")):
                        if len(dialogue_line_text) > 1:
                           dialogue_line_text = dialogue_line_text[1:-1].strip()

                    dialogue_entry = {
                        "Speaker_Slug": final_subject_slug,
                        "Parenthetical": None,
                        "Line": dialogue_line_text,
                        "Is_Voice_Over": False
                    }
                    
                    # Transfer attributes mined during analysis
                    if 'attributes' in beat and final_subject_slug in beat['attributes']:
                         attrs = beat['attributes'][final_subject_slug]
                         # Filter empty ones
                         final_attrs = {k: v for k, v in attrs.items() if v}
                         if final_attrs:
                              dialogue_entry['attributes'] = final_attrs
                         # Remove the attributes from the beat level as they are now in the dialogue line
                         del beat['attributes']

                    # Transfer instructions if any
                    if 'instructions' in beat:
                         dialogue_entry['instructions'] = beat['instructions']
                         del beat['instructions']

                    beat['dialogue_lines'] = [dialogue_entry]


            # B. Aggregate Attributes (All Beat Types)
            
            # B.1 Narrative/Structure (Attributes derived during phase A)
            if 'attributes' in beat and beat.get('attributes'):
                for slug, attrs in beat['attributes'].items():
                    active_agents.add(slug)
                    current_shot_subjects.add(slug)
                    
                    if slug not in shot_attributes: shot_attributes[slug] = {}
                    for attr_type, attr_list in attrs.items():
                         shot_attributes[slug].setdefault(attr_type, []).extend(attr_list)


            # B.2 Dialogue (including V15.4 re-classified beats)
            if beat['type'] == 'dialogue':
                for line in beat.get('dialogue_lines', []):
                    slug = line['Speaker_Slug']
                    if slug:
                        active_agents.add(slug)
                        current_shot_subjects.add(slug)
                        
                        # Attributes from parentheticals or implicit analysis
                        if 'attributes' in line:
                            if slug not in shot_attributes: shot_attributes[slug] = {}
                            for attr_type, attrs in line['attributes'].items():
                                shot_attributes[slug].setdefault(attr_type, []).extend(attrs)
                    
                    # Collect Instructions
                    if 'instructions' in line:
                        for k, v in line['instructions'].items():
                            shot_instructions.setdefault(k, []).extend(v)
            

            # C. Instructions (General)
            if 'instructions' in beat and beat.get('instructions'):
                 for k, v in beat['instructions'].items():
                    shot_instructions.setdefault(k, []).extend(v)
            
            processed_beats.append(beat)

        # V15.4: Update shot beats
        shot['Beats'] = processed_beats

        # Update Recent Active Subject (Stable Logic)
        individual_subjects = {s for s in current_shot_subjects if s in G_CHARS_BY_SLUG}
        
        if len(individual_subjects) == 1:
            recent_active_subject = list(individual_subjects)[0]
        elif len(individual_subjects) > 1:
            recent_active_subject = None
        elif len(current_shot_subjects) == 1:
             recent_active_subject = list(current_shot_subjects)[0]
        elif len(current_shot_subjects) > 1:
             recent_active_subject = None
        # If 0 subjects, recent_active_subject remains unchanged.


        # Update Scene Presence
        scene_presence.update(active_agents)

        # --- 4.2: Update Persistence Tracker (V15.1 Logic) ---
        for slug, attrs in shot_attributes.items():
            
            if slug not in persistence_tracker:
                # V15.1 Initialize
                persistence_tracker[slug] = {"Appearance": [], "Emotions": [], "Stance": None, "Posture_Modifiers": []}

            current_dynamic_actions = attrs.get('Actions', [])

            # 1. Update Appearance
            if 'Appearance' in attrs or current_dynamic_actions:
                negations = rules.get("attribute_negations", {})
                for action in current_dynamic_actions:
                    if action in negations:
                        for negated_attr in negations[action]:
                            if negated_attr in persistence_tracker[slug]['Appearance']:
                                persistence_tracker[slug]['Appearance'].remove(negated_attr)
                
                for app in attrs.get('Appearance', []):
                    if app not in persistence_tracker[slug]['Appearance']:
                        persistence_tracker[slug]['Appearance'].append(app)
                persistence_tracker[slug]['Appearance'].sort()

            # 2. Update Emotions
            if 'Emotions' in attrs and attrs['Emotions']:
                current_emotions = list(persistence_tracker[slug]['Emotions'])
                new_emotions = attrs['Emotions']
                resolved_emotions = resolve_emotion_conflicts(current_emotions, new_emotions, rules.get("emotion_conflict_groups", []))
                persistence_tracker[slug]['Emotions'] = resolved_emotions

            # 3. Update Stance (V15.1)
            current_stance = persistence_tracker[slug]['Stance']
            new_stance = attrs.get('Stance', [None])[0] if attrs.get('Stance') else None
            
            resolved_stance = resolve_stance_conflicts(
                current_stance, new_stance, current_dynamic_actions, rules.get("stance_conflict_map", {})
            )
            persistence_tracker[slug]['Stance'] = resolved_stance

            # 4. Update Posture Modifiers (V15.1)
            current_modifiers = list(persistence_tracker[slug]['Posture_Modifiers'])
            new_modifiers = attrs.get('Posture_Modifiers', [])
            
            resolved_modifiers = resolve_modifier_conflicts(
                current_modifiers, new_modifiers, current_dynamic_actions, rules.get("modifier_conflict_map", {})
            )
            persistence_tracker[slug]['Posture_Modifiers'] = resolved_modifiers


        # --- 4.3: Finalize Shot Composition (V15.1 Structure) ---
        
        # V15.0: Iterate over scene_presence (includes inter-scene persistence)
        for slug in scene_presence:
            is_group = slug in G_GROUPS_BY_SLUG
            comp_key = "Groups" if is_group else "Characters"

            is_active = slug in active_agents
            status = "active" if is_active else "passive"

            shot_composition[comp_key][slug] = {"Status": status, "Attributes": {}}
            
            # A. Apply Persistent State (V15.1)
            if slug in persistence_tracker:
                # Appearance, Emotions, Posture_Modifiers (Lists)
                for attr_type in ["Appearance", "Emotions", "Posture_Modifiers"]:
                    attrs = persistence_tracker[slug].get(attr_type, [])
                    if attrs:
                        shot_composition[comp_key][slug]["Attributes"][attr_type] = sorted(list(set(attrs)))
                
                # V15.1: Stance (Single value, stored as a list for consistency)
                stance = persistence_tracker[slug].get('Stance')
                if stance:
                     shot_composition[comp_key][slug]["Attributes"]["Stance"] = [stance]


            # B. Apply Immediate State
            if slug in shot_attributes:
                 for attr_type, attrs in shot_attributes[slug].items():
                     if attrs:
                        if attr_type not in shot_composition[comp_key][slug]["Attributes"]:
                             shot_composition[comp_key][slug]["Attributes"][attr_type] = []
                        
                        # Merge immediate attributes on top
                        existing = set(shot_composition[comp_key][slug]["Attributes"][attr_type])
                        for attr in attrs:
                            if attr not in existing:
                                shot_composition[comp_key][slug]["Attributes"][attr_type].append(attr)
                        shot_composition[comp_key][slug]["Attributes"][attr_type].sort()


            # C. Apply Default Passive States (V15.1 Update)
            if status == "passive":
                attrs = shot_composition[comp_key][slug]["Attributes"]
                
                # V15.1: Check Stance default
                if not attrs.get("Stance") and rules.get("default_passive_stance"):
                     attrs.setdefault("Stance", []).append(rules["default_passive_stance"])

                if not attrs.get("Actions") and rules.get("default_passive_action"):
                    attrs.setdefault("Actions", []).append(rules["default_passive_action"])
                
                if not attrs.get("Emotions") and rules.get("default_passive_emotion"):
                     attrs.setdefault("Emotions", []).append(rules["default_passive_emotion"])

        
        # Attribute Deduplication (Stable)
        for slug, data in shot_composition["Characters"].items():
            global_char_data = G_CHARS_BY_SLUG.get(slug)
            if global_char_data and 'Attributes' in data:
                data['Attributes'] = filter_inherent_attributes(data['Attributes'], global_char_data['inherent_attributes'])
                if not data['Attributes']:
                     del data['Attributes']
        
        for slug, data in shot_composition["Groups"].items():
             global_group_data = G_GROUPS_BY_SLUG.get(slug)
             if global_group_data and 'Attributes' in data:
                data['Attributes'] = filter_inherent_attributes(data['Attributes'], global_group_data.get('inherent_attributes', {}))
                if not data['Attributes']:
                     del data['Attributes']


        # Finalize Instructions
        final_instructions = {}
        for instr_type, instr_list in shot_instructions.items():
            if instr_list:
                final_instructions[instr_type] = sorted(list(set(instr_list)))

        # Add finalized data to the shot
        shot["Shot_Composition"] = shot_composition
        shot["Shot_Type"] = determine_shot_type(shot_composition, final_instructions)
        if final_instructions:
            shot["Instructions"] = final_instructions

    # Assemble the final scene dictionary
    final_scene = {
        "Scene_ID": scene_data["Scene_ID"],
        "Title": scene_data["Title"],
        "Setting": context["setting"],
        "TimeOfDay": context["time_of_day"],
        "Tone": context["tone"],
        "Props": context["props"],
        "Shots": shots
    }
    return final_scene

# ===========================================================
# SECTION 6: MAIN EXECUTION (V15.0, V15.2 Updates)
# ===========================================================

def normalize_script(script_input:str, profile_name="PIXAR_3D_VI", title=None, is_file=True, output_dir="./output_normalized_v15_final"):
    
    # 1. Initialize Context
    try:
        ctx = initialize_context(profile_name)
    except ValueError as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return None

    # 2. Load Script Content
    if is_file:
        try:
            script_path = Path(script_input).resolve()
            script_content = script_path.read_text(encoding="utf-8")
            if title is None:
                title = script_path.stem
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy file kịch bản tại {script_input}")
            return None
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {e}")
            return None
    else:
        script_content = script_input
        if title is None:
            title = "untitled_script"

    # 3. Pass 1: Global Character Consolidation (V15.2 Applied)
    try:
        # V15.2: Updated return values
        global_map, global_characters, global_group_agents, local_anaphora_terms, role_map = global_character_pass(script_content, ctx)
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng trong quá trình Global Pass (Pass 1): {e}")
        # import traceback; traceback.print_exc()
        return None

    # 4. Prepare Global Context
    G_CHARS_BY_SLUG = {data['slug']: data for data in global_characters.values()}
    G_GROUPS_BY_SLUG = {data['slug']: data for data in global_group_agents.values()}

    # Unified lookup map (Includes Global Anaphora)
    agent_slug_lookup = {}
    for term, c_name in global_map.items():
        if c_name in global_characters:
            agent_slug_lookup[canonicalize(term).lower()] = global_characters[c_name]['slug']
    
    for c_name, data in global_characters.items():
         agent_slug_lookup[canonicalize(c_name).lower()] = data['slug']
            
    for g_name, data in global_group_agents.items():
        agent_slug_lookup[canonicalize(g_name).lower()] = data['slug']

    # V15.2: Enhanced Global Context
    GLOBAL_CONTEXT = {
        'Global_Characters': global_characters,
        'Global_Group_Agents': global_group_agents,
        'Global_Characters_By_Slug': G_CHARS_BY_SLUG,
        'Global_Group_Agents_By_Slug': G_GROUPS_BY_SLUG,
        'Agent_Slug_Lookup': agent_slug_lookup,
        'Local_Anaphora_Terms': local_anaphora_terms,
        'Role_Map': role_map
    }

    # 5. Scene Detection
    scenes_data = detect_scenes(script_content, ctx)
    
    # 6. Pass 2: Scene Processing (V15.0 Architecture)
    processed_scenes = []
    
    # V15.0: Initialize Global Persistence Tracker
    # This tracker is passed sequentially through all scenes.
    global_persistence_tracker = {} 

    try:
        for sc_data in scenes_data:
            # V15.0: Pass the tracker by reference
            processed_scene = process_scene(sc_data, GLOBAL_CONTEXT, ctx, global_persistence_tracker)
            processed_scenes.append(processed_scene)
    except Exception as e:
        import traceback
        print(f"❌ Lỗi nghiêm trọng trong quá trình Scene Processing (Pass 2) tại Cảnh {sc_data.get('Scene_ID', 'N/A')}: {e}")
        traceback.print_exc()
        return None

    # 7. Assemble StoryGrid
    sorted_global_chars = dict(sorted(global_characters.items(), key=lambda item: item[1]['slug']))
    sorted_global_groups = dict(sorted(global_group_agents.items(), key=lambda item: item[1]['slug']))

    story={
        "Project":{
            "Title": title,
            "Language": ctx['profile']['language'],
            "Genre": ctx['profile']['genre'],
            "StoryGrid_Version": ctx['profile']['StoryGrid_Version']
        },
        "Global_Characters": sorted_global_chars,
        "Global_Group_Agents": sorted_global_groups,
        "Scenes": processed_scenes
    }

    # 8. Output results
    if is_file:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_json_path = Path(output_dir, f"storygrid_v15_{title}_FINAL.json")
        
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(story, f, ensure_ascii=False, indent=2)
            print(f"✅ Hoàn tất (V15 FINAL): Phân tích thành công {len(processed_scenes)} cảnh (Production-Ready).")
            print(f"🧾 Kết quả xuất tại: {output_json_path}")
        except Exception as e:
            print(f"❌ Lỗi khi ghi file JSON: {e}")

    return story

# ---------- CLI ----------
if __name__=="__main__":
    # Set stdout encoding to UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
        except Exception as e:
            print(f"Cảnh báo: Không thể thiết lập mã hóa UTF-8: {e}")

    ap=argparse.ArgumentParser(description="Screenplay Normalizer (v15.0 FINAL) → StoryGrid v5.0")
    ap.add_argument("--script", required=False, help="Đường dẫn file kịch bản .txt")
    ap.add_argument("--profile", default="PIXAR_3D_VI", help="Tên profile cấu hình (mặc định: PIXAR_3D_VI)")
    ap.add_argument("--output", default="./output_normalized_v15_final", help="Thư mục xuất kết quả")
    
    try:
        # Basic check if running interactively or via CLI
        if len(sys.argv) > 1:
             args=ap.parse_args()
             if args.script:
                normalize_script(args.script, profile_name=args.profile, output_dir=args.output, is_file=True)
             else:
                 print("Lỗi: Thiếu tham số --script.")
        elif not 'ipykernel' in sys.modules and sys.stdin.isatty():
             print("Chạy ở chế độ CLI yêu cầu tham số --script. Sử dụng --help để xem hướng dẫn.")

    except SystemExit:
        pass