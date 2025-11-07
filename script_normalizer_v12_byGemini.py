# ===========================================================
# script_normalizer_v12.py
# Screenplay Normalizer → StoryGrid v4.0 (Data Integrity & Semantic Accuracy)
# - V12.1: Emotion Conflict Resolution (Ensures logical emotional states).
# - V12.2: Guaranteed Scene Presence (Fixes disappearing passive characters).
# - V12.3: Robust Parenthetical Parsing (Fixes complex/nested/concatenation errors).
# - V12.4: Attribute Deduplication Fix (Ensures only transient attributes in shots).
# - V12.5: Initial State Injection (Analyzes Setting line for starting conditions).
# - V12.7: Centralized Context Management (Optimization).
# ===========================================================
import re, json, argparse, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
import itertools
import sys
import copy

# Import the profile loader
try:
    import profiles_v12 as profiles
except ImportError:
    print("❌ Lỗi: Không tìm thấy module 'profiles_v12.py'.")
    sys.exit(1)

# ===========================================================
# SECTION 1: UTILITIES & HELPERS
# ===========================================================

# (Utilities nfc, canonicalize, strip_combining, slugify, clean_title remain identical)
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
    """(V12.3) Extracts parentheticals (handling nesting) and returns the clean text."""
    parentheticals = []; cleaned_text_parts = []
    nesting_level = 0; last_extraction_end = 0; start_index = 0

    for i, char in enumerate(text):
        if char == '(':
            if nesting_level == 0:
                cleaned_text_parts.append(text[last_extraction_end:i])
                start_index = i
            nesting_level += 1
        elif char == ')':
            if nesting_level > 0:
                nesting_level -= 1
                if nesting_level == 0:
                    content = text[start_index+1:i].strip()
                    if content:
                        parentheticals.append(content)
                    last_extraction_end = i + 1

    if last_extraction_end < len(text):
        cleaned_text_parts.append(text[last_extraction_end:])

    cleaned_text = "".join(cleaned_text_parts).strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # V12.3: Final cleanup of potential trailing/leading punctuation/quotes
    cleaned_text = cleaned_text.strip('.,?!"')
    return parentheticals, cleaned_text


# ===========================================================
# SECTION 2: DYNAMIC PROFILE SYSTEM (V12.7 Optimization)
# ===========================================================

# V12.7: Centralized Regex Compilation
def initialize_context(profile_name: str) -> Dict[str, Any]:
    """Loads the profile, compiles regexes, and prepares the execution context."""
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
    LOWERCASE_GROUP = "(?:" + "|".join(re.escape(w) for w in ALLOWED_LOWERCASE_WORDS) + ")"

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
# SECTION 3: CORE LOGIC - PARSING & EXTRACTION
# ===========================================================

# (Validation Helpers, Phrase Mining Helpers, Scene Detection, Global Character Pass remain stable, utilizing the centralized Context)

# ---------- Validation Helpers ----------

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


# --- Phrase Mining Helpers ---

def mine_phrases(text: str, phrase_list: List[str]) -> List[str]:
    found = set()
    sorted_phrases = sorted(phrase_list, key=len, reverse=True)
    canonical_text = canonicalize(text)

    for phrase in sorted_phrases:
        canonical_phrase = canonicalize(phrase)
        if re.search(rf"\b{re.escape(canonical_phrase)}\b", canonical_text, flags=re.I):
            is_subsumed = False
            for existing in found:
                if re.search(rf"\b{re.escape(canonical_phrase)}\b", canonicalize(existing), flags=re.I):
                    is_subsumed = True; break
            if not is_subsumed:
                to_remove = set()
                for existing in found:
                    if re.search(rf"\b{re.escape(canonicalize(existing))}\b", canonical_phrase, flags=re.I):
                        to_remove.add(existing)
                found -= to_remove
                found.add(phrase)
    return sorted(list(found))

def mine_appearance(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons']['appearance_phrases'])
def mine_actions(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons']['action_phrases'])
def mine_emotions(text: str, ctx: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, ctx['lexicons']['emotion_phrases'])

def mine_cinematic_instructions(text: str, ctx: Dict[str, Any]) -> Dict[str, List[str]]:
    if not text: return {}
    instructions = {"camera": [], "vfx_sfx": [], "meta": []}
    config = ctx['cinematic']
    
    for keyword, tag in config.get('camera_moves', {}).items():
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.I): instructions["camera"].append(tag)
    for keyword, tag in config.get('vfx_sfx', {}).items():
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.I): instructions["vfx_sfx"].append(tag)
    for pattern, tag in config.get('meta_types', {}).items():
        if re.search(pattern, text, re.I): instructions["meta"].append(tag)
            
    return {k: sorted(list(set(v))) for k, v in instructions.items() if v}


# ---------- Scene Detection ----------
def detect_scenes(text:str, ctx: Dict[str, Any])->List[Dict[str,Any]]:
    SCENE_HEADER_PATS = ctx['rules']['scene_header_patterns']
    canonical_text = canonicalize(text); lines = canonical_text.splitlines(); idxs=[]
    for i,L in enumerate(lines):
        if any(re.search(p, L.strip(), flags=re.I) for p in SCENE_HEADER_PATS): idxs.append(i)

    if not idxs: return [{"Scene_ID":1,"Title":"Untitled","Raw":canonical_text}]

    idxs.append(len(lines)); out=[]
    for si,(a,b) in enumerate(zip(idxs, idxs[1:]), start=1):
        title = clean_title(lines[a].strip())
        body  = "\n".join(lines[a+1:b]).strip()
        if body: out.append({"Scene_ID":si,"Title":title,"Raw":body})
    return out

# ---------- Global Pass: Character Consolidation ----------

def global_character_pass(full_text:str, ctx: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, Dict]]:
    # (Implementation remains stable, utilizing the centralized Context)
    text = canonicalize(full_text)
    raw_terms = set(); explicit_aliases = {}
    regexes = ctx['regexes']; rules = ctx['rules']

    def process_term(name, alias=None):
        is_valid, cleaned_name = validate_and_clean_name(name, ctx, allow_group_agents=False)
        if is_valid:
            raw_terms.add(cleaned_name)
            if alias:
                is_valid_a, cleaned_alias = validate_and_clean_name(alias, ctx, allow_group_agents=False)
                if is_valid_a:
                     raw_terms.add(cleaned_alias); explicit_aliases[cleaned_alias] = cleaned_name

    # 1. Extraction
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

    # 2. Resolution & 3. Consolidation & 4. Final Mapping
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
            if name_b.lower().startswith(name_a.lower() + " "):
                canonical_map[name_a] = canonical_map[name_b]

    global_term_to_canonical_map = {}
    for term in raw_terms:
        resolved = term
        for alias, name in explicit_aliases.items():
            if alias.lower() == term.lower(): resolved = name; break
        
        canonical = None
        for variation, canon in canonical_map.items():
            if variation.lower() == resolved.lower(): canonical = canon; break

        if canonical: global_term_to_canonical_map[term] = canonical

    # 5. Building Global Character Registry
    global_characters = {}
    canonical_names = set(canonical_map.values())
    
    for c_name in canonical_names:
        role = "student"
        if any(t in c_name.lower() for t in rules['role_self_teacher']): role = "teacher"
        
        aliases = []
        for term, canon in global_term_to_canonical_map.items():
            if canon == c_name and term != c_name: aliases.append(term)

        inherent_appearance = mine_appearance(c_name, ctx)

        global_characters[c_name] = {
            "role": role,
            "aliases": sorted(aliases),
            "slug": slugify(c_name),
            "inherent_attributes": {
                "Appearance": inherent_appearance,
                "Actions": [],
                "Emotions": []
            }
        }

    # 6. Building Global Group Agent Registry
    global_group_agents = {}
    for agent_name in rules.get('group_agents', []):
        if re.search(rf"\b{re.escape(agent_name)}\b", text, flags=re.I):
             # Ensure structure consistency
             inherent_appearance = mine_appearance(agent_name, ctx)
             global_group_agents[agent_name] = {
                "slug": slugify(agent_name),
                "type": "group",
                "inherent_attributes": { 
                    "Appearance": inherent_appearance,
                    "Actions": [],
                    "Emotions": []
                }
            }

    return global_term_to_canonical_map, global_characters, global_group_agents

# ---------- Context Derivation ----------

def derive_context(scene_text:str, ctx: Dict[str, Any])->Dict[str,Any]:
    # (Implementation remains stable, updated to return setting_line for V12.5)
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

    # Prop consolidation logic remains stable
    low = s.lower(); raw_props = set()
    sorted_props_list = sorted(lexicons['props_list'], key=len, reverse=True)

    for kw in sorted_props_list:
        canonical_kw = canonicalize(kw)
        if re.search(rf"\b{re.escape(canonical_kw)}\b", low, re.I):
            raw_props.add(kw)

    sorted_raw_props = sorted(list(raw_props), key=len); final_props = set(raw_props)
    for i, prop_a in enumerate(sorted_raw_props):
        for prop_b in sorted_raw_props[i+1:]:
            if prop_a in prop_b:
                 if prop_a in final_props:
                     try: final_props.remove(prop_a)
                     except KeyError: pass
                 break

    # TOD and Tone logic remains stable
    tod = "morning"
    if ctx['profile']['language'] == 'vi' and ("sáng sớm" in low or "nắng sớm" in low): 
        tod = "early_morning"

    tone=[];
    for key, val in lexicons['tone_map']:
        if key in low: tone.append(val)

    if not tone: tone=["warm","gentle"]

    # V12.5: Return setting_line as well
    return {"setting":setting,"props":sorted(list(final_props)),"time_of_day":tod,"tone":sorted(list(set(tone))), "setting_line": setting_line}


# ===========================================================
# SECTION 4: BEAT & SHOT PROCESSING (V12.3 Updates)
# ===========================================================

# (Line Classification, safe_clean_labeled_content remain stable)
LINE_TYPES = {
    "DIALOGUE": "dialogue", "ACTION": "action", "STRUCTURE": "structure",
    "META": "meta", "IGNORE": "ignore"
}

def classify_line(line:str, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
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

    for label in rules['structure_labels'].keys():
        pattern = rf"^(?:[\*\-•]\s*)?{re.escape(label)}\s*(?:\(.*\))?\s*:\s*"
        if re.match(pattern, clean_line, re.I):
            cleaned = re.sub(pattern, "", clean_line, count=1, flags=re.I).strip()
            return cleaned
    return line.strip()

# ---------- Narrative Analysis (Bidirectional Search) ----------

def analyze_narrative_attributes(text_lines: List[str], agent_slug_lookup: Dict[str, str], ctx: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    # (Implementation remains stable, utilizing the centralized Context and updated Lexicons)
    attributes_by_slug = {}
    regexes = ctx['regexes']; rules = ctx['rules']; lexicons = ctx['lexicons']
    NEGATION_KEYWORDS = rules.get('negation_keywords', [])

    full_text = " ".join(text_lines)
    sentences = regexes['SENTENCE_SPLIT_RE'].split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Handle case with no punctuation
    if not sentences and full_text.strip():
        sentences = [full_text.strip()]

    sorted_terms = sorted(agent_slug_lookup.keys(), key=len, reverse=True)
    sorted_actions = sorted(lexicons['action_phrases'], key=len, reverse=True)
    sorted_emotions = sorted(lexicons['emotion_phrases'], key=len, reverse=True)
    sorted_appearance = sorted(lexicons['appearance_phrases'], key=len, reverse=True)

    def initialize_slug(slug):
        if slug not in attributes_by_slug:
            attributes_by_slug[slug] = {"Actions": [], "Emotions": [], "Appearance": []}

    def find_entities(sentence, terms, type_label):
        entities = []; canonical_sentence = canonicalize(sentence)
        for term in terms:
            canonical_term = canonicalize(term)
            for match in re.finditer(rf"\b{re.escape(canonical_term)}\b", canonical_sentence, re.I):
                entity_value = term
                if type_label == "AGENT":
                     entity_value = agent_slug_lookup.get(term.lower())
                
                if entity_value:
                    is_overlapped = False
                    for existing in entities:
                        if existing['type'] == type_label and match.start() >= existing['start'] and match.end() <= existing['end']:
                            is_overlapped = True; break
                    if not is_overlapped:
                        entities.append({"start": match.start(), "end": match.end(), "value": entity_value, "type": type_label})
        return entities

    for sentence in sentences:
        agents = find_entities(sentence, sorted_terms, "AGENT")
        actions = find_entities(sentence, sorted_actions, "ACTION")
        emotions = find_entities(sentence, sorted_emotions, "EMOTION")
        appearances = find_entities(sentence, sorted_appearance, "APPEARANCE")

        all_entities = sorted(agents + actions + emotions + appearances, key=lambda x: x['start'])

        # Bidirectional Proximity Association
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
                is_negated = any(re.search(rf"\b{re.escape(kw)}\b", segment, re.I) for kw in NEGATION_KEYWORDS)

                if not is_negated:
                    attr_type_map = {"ACTION": "Actions", "EMOTION": "Emotions", "APPEARANCE": "Appearance"}
                    attr_key = attr_type_map.get(next_entity['type'])
                    if attr_key and next_entity['value'] not in attributes_by_slug[slug][attr_key]:
                        attributes_by_slug[slug][attr_key].append(next_entity['value'])

            # Search Backward (Left)
            for j in range(i - 1, -1, -1):
                prev_entity = all_entities[j]
                if prev_entity['type'] == "AGENT": break

                segment = sentence[prev_entity['end']:entity['start']]
                is_negated = any(re.search(rf"\b{re.escape(kw)}\b", segment, re.I) for kw in NEGATION_KEYWORDS)

                if not is_negated:
                    # Primarily allow Emotions/States preceding the agent
                    if prev_entity['type'] == "EMOTION":
                        if prev_entity['value'] not in attributes_by_slug[slug]["Emotions"]:
                            attributes_by_slug[slug]["Emotions"].append(prev_entity['value'])
    
    # Sort results
    for slug in attributes_by_slug:
        for key in attributes_by_slug[slug]:
            attributes_by_slug[slug][key] = sorted(attributes_by_slug[slug][key])

    return attributes_by_slug


# ---------- Beat Extraction (V12.3 Update) ----------

def extract_beats(scene_text:str, agent_slug_lookup: Dict[str, str], ctx: Dict[str, Any]) -> List[Dict[str,Any]]:
    """(V12.3) Segments the scene, ensuring robust dialogue parsing."""
    lines = scene_text.splitlines()
    classified_lines = []
    for line in lines:
        l_type, content, tag = classify_line(line, ctx)
        if l_type != LINE_TYPES["IGNORE"]:
            classified_lines.append({"type": l_type, "raw": content, "tag": tag})

    if not classified_lines:
        return [] 

    beats = []; beat_id = 1; regexes = ctx['regexes']

    for group_type, group_iter in itertools.groupby(classified_lines, key=lambda x: x['type']):
        group = list(group_iter)

        if group_type == LINE_TYPES["STRUCTURE"]:
            # (Logic remains stable)
            for item in group:
                tag = item['tag']
                if tag == 'setting': continue
                cleaned_text = safe_clean_labeled_content(item['raw'], ctx)
                if cleaned_text:
                    beat = {"id":f"B{beat_id}","type":tag,"text_lines":[cleaned_text]}
                    if tag in ['arrival', 'climax', 'conclusion']:
                        attrs = analyze_narrative_attributes([cleaned_text], agent_slug_lookup, ctx)
                        if attrs: beat["attributes"] = attrs
                        instructions = mine_cinematic_instructions(cleaned_text, ctx)
                        if instructions: beat["instructions"] = instructions
                    
                    beats.append(beat); beat_id += 1

        elif group_type == LINE_TYPES["DIALOGUE"]:
            # V12.3 Implementation
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
                    speaker_slug = agent_slug_lookup.get(canonical_speaker, slugify(clean_speaker_name))

                    # V12.3: Robust extraction and cleaning
                    parentheticals_raw, clean_line = extract_robust_parentheticals(dialogue_content)
                    
                    # V12.3: Use "; " separator for raw text representation
                    parenthetical_text = "; ".join(parentheticals_raw) if parentheticals_raw else None

                    # Analysis
                    instructions = {}
                    attributes = {"Actions": [], "Emotions": [], "Appearance": []}
                    
                    # Process EACH parenthetical independently
                    for p_text in parentheticals_raw:
                        attributes["Actions"].extend(mine_actions(p_text, ctx))
                        attributes["Emotions"].extend(mine_emotions(p_text, ctx))
                        attributes["Appearance"].extend(mine_appearance(p_text, ctx))
                        
                        p_instr = mine_cinematic_instructions(p_text, ctx)
                        for k, v in p_instr.items():
                            instructions.setdefault(k, []).extend(v)

                    # V.O. Detection
                    is_vo = False
                    if "(V.O)" in speaker_raw or "(V.O.)" in speaker_raw or "lồng tiếng" in speaker_raw.lower():
                        is_vo = True

                    dialogue_entry = {
                        "Speaker_Slug": speaker_slug,
                        "Parenthetical": parenthetical_text,
                        "Line": clean_line,
                        "Is_Voice_Over": is_vo
                    }
                    
                    # Final cleanup and addition
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
            # (Logic remains stable)
            text_lines = [item['raw'] for item in group]
            beat_type = "establish" if (beat_id == 1 and not beats and group_type == LINE_TYPES["ACTION"]) else group[0]['tag'] or group_type.lower()

            beat = {"id":f"B{beat_id}","type":beat_type,"text_lines":text_lines}

            if group_type == LINE_TYPES["ACTION"]:
                # Clean labeled action lines before analysis
                if group[0]['tag'] == 'action':
                     text_lines_for_analysis = [safe_clean_labeled_content(line, ctx) for line in text_lines]
                else:
                     text_lines_for_analysis = text_lines

                attrs = analyze_narrative_attributes(text_lines_for_analysis, agent_slug_lookup, ctx)
                if attrs: beat["attributes"] = attrs

            instructions = mine_cinematic_instructions(" ".join(text_lines), ctx)
            if instructions: beat["instructions"] = instructions

            beats.append(beat); beat_id += 1

    # Post-processing for initial beat type
    if beats and beats[0]['type'] == 'action':
         beats[0]['type'] = 'establish'

    return beats

# ===========================================================
# SECTION 5: SCENE PROCESSING & CINEMATIC LOGIC (V12.1, V12.2, V12.4, V12.5 Updates)
# ===========================================================

# (Shot Segmentation and determine_shot_type remain stable)
def segment_shots(beats: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots = []; shot_id = 1; regexes = ctx['regexes']

    for beat in beats:
        if beat['type'] in ['dialogue', 'meta', 'setting']:
            shots.append({"Shot_ID": f"S{shot_id}", "Beats": [beat]}); shot_id += 1
        else:
            # Split narrative beats by sentence (V11 logic maintained)
            text = " ".join(beat['text_lines'])
            sentences = regexes['SENTENCE_SPLIT_RE'].split(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                shots.append({"Shot_ID": f"S{shot_id}", "Beats": [beat]}); shot_id += 1
            else:
                # Create sub-beats for each sentence
                for i, sentence in enumerate(sentences):
                    sub_beat = copy.deepcopy(beat)
                    sub_beat['id'] = f"{beat['id']}.{i+1}"
                    sub_beat['text_lines'] = [sentence]
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

# V12.1: Helper for Emotion Conflict Resolution
def resolve_emotion_conflicts(current_emotions: List[str], new_emotions: List[str], conflict_groups: List[Dict[str, Any]]) -> List[str]:
    """Resolves conflicts between existing and new emotions, prioritizing new ones."""
    if not new_emotions or not conflict_groups:
        # If no new emotions or no conflict rules, just combine and deduplicate
        return sorted(list(set(current_emotions + new_emotions)))

    resolved = list(current_emotions)
    
    for new_emotion in new_emotions:
        conflicts_to_remove = set()
        # Check if the new emotion causes conflicts with existing ones
        for group in conflict_groups:
            if new_emotion in group['emotions']:
                for existing_emotion in resolved:
                    if existing_emotion in group['emotions'] and existing_emotion != new_emotion:
                        conflicts_to_remove.add(existing_emotion)
        
        # Remove conflicts
        resolved = [e for e in resolved if e not in conflicts_to_remove]
        
        # Add new emotion if not present
        if new_emotion not in resolved:
            resolved.append(new_emotion)

    return sorted(resolved)


# V12.4: Helper function for Robust Attribute Deduplication
def filter_inherent_attributes(attributes: Dict[str, List[str]], inherent_attributes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Removes inherent attributes using normalized comparison."""
    filtered = copy.deepcopy(attributes)
    
    # Normalize inherent appearance for robust comparison
    inherent_appearance_set = set(canonicalize(a).lower() for a in inherent_attributes.get("Appearance", []))
    
    if "Appearance" in filtered:
        # Filter: keep if normalized version is not in inherent_set
        filtered_appearance = [
            attr for attr in filtered["Appearance"] 
            if canonicalize(attr).lower() not in inherent_appearance_set
        ]
        
        if filtered_appearance:
            filtered["Appearance"] = sorted(filtered_appearance)
        else:
            del filtered["Appearance"]
            
    return filtered

# ---------- Scene Processing (V12 Main Logic) ----------

def process_scene(scene_data: Dict[str, Any], global_context: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    (V12) Main scene processing pipeline: Context, Beats, Shots, Composition, Persistence.
    """
    G_CHARS_BY_SLUG = global_context['Global_Characters_By_Slug']
    G_GROUPS_BY_SLUG = global_context['Global_Group_Agents_By_Slug']
    AGENT_SLUG_LOOKUP = global_context['Agent_Slug_Lookup']
    rules = ctx['rules']

    # 1. Context Derivation
    context = derive_context(scene_data['Raw'], ctx)

    # 2. Beat Extraction (V12.3 Parsing Fixes Applied here)
    beats = extract_beats(scene_data['Raw'], AGENT_SLUG_LOOKUP, ctx)

    # 3. Shot Segmentation
    shots = segment_shots(beats, ctx)

    # 4. Shot Composition & Attribute Persistence Initialization
    persistence_tracker = {} # Slug -> {Appearance: [], Emotions: []}
    # V12.2: Track physical presence in the scene
    scene_presence = set() 

    # V12.5: Initial State Injection (Analyze Setting Line)
    if context.get("setting_line"):
        # Analyze the setting line using the narrative analysis logic
        initial_attrs = analyze_narrative_attributes([context["setting_line"]], AGENT_SLUG_LOOKUP, ctx)
        for slug, attrs in initial_attrs.items():
            # Mark as present from the start
            scene_presence.add(slug)
            if slug not in persistence_tracker:
                persistence_tracker[slug] = {"Appearance": [], "Emotions": []}
            
            # Initialize state (Actions are immediate, Appearance/Emotions persist)
            if 'Appearance' in attrs: 
                persistence_tracker[slug]['Appearance'] = attrs['Appearance']
            if 'Emotions' in attrs: 
                # Apply conflict resolution even at initialization
                persistence_tracker[slug]['Emotions'] = resolve_emotion_conflicts([], attrs['Emotions'], rules.get("emotion_conflict_groups", []))
            # Note: Actions from the setting are handled during the first shot processing if needed, 
            # but typically setting describes initial persistent states.


    for shot in shots:
        shot_composition = {"Characters": {}, "Groups": {}}
        shot_instructions = {}
        active_agents = set()
        shot_attributes = {} # Attributes specific to this shot (immediate actions/emotions/appearance)

        # --- 4.1: Process Beats within the Shot (Aggregate Data) ---
        for beat in shot['Beats']:
            
            # Re-analyze attributes if they were split during segmentation (sub-beats)
            if '.' in beat['id'] and beat['type'] not in ['dialogue', 'meta', 'setting']:
                 beat_attrs = analyze_narrative_attributes(beat['text_lines'], AGENT_SLUG_LOOKUP, ctx)
                 beat['attributes'] = beat_attrs # Update the sub-beat data
                 beat['instructions'] = mine_cinematic_instructions(" ".join(beat['text_lines']), ctx)

            # A. Narrative/Structure Beats
            if 'attributes' in beat:
                for slug, attrs in beat['attributes'].items():
                    active_agents.add(slug)
                    shot_attributes.setdefault(slug, {}).update(attrs)


            # B. Dialogue Beats
            if beat['type'] == 'dialogue':
                for line in beat.get('dialogue_lines', []):
                    slug = line['Speaker_Slug']
                    if slug:
                        active_agents.add(slug)
                        # Use structured attributes
                        if 'attributes' in line:
                            if slug not in shot_attributes: shot_attributes[slug] = {}
                            # Merge attributes
                            for attr_type, attrs in line['attributes'].items():
                                shot_attributes[slug].setdefault(attr_type, []).extend(attrs)
                    
                    # Collect Instructions
                    if 'instructions' in line:
                        for k, v in line['instructions'].items():
                            shot_instructions.setdefault(k, []).extend(v)

            # C. Instructions (non-dialogue)
            if 'instructions' in beat:
                 for k, v in beat['instructions'].items():
                    shot_instructions.setdefault(k, []).extend(v)

        # V12.2: Update Scene Presence
        scene_presence.update(active_agents)

        # --- 4.2: Update Persistence Tracker based on Shot Attributes ---
        for slug, attrs in shot_attributes.items():
            
            if slug not in persistence_tracker:
                persistence_tracker[slug] = {"Appearance": [], "Emotions": []}

            # Update Persistence State: Appearance (Merge and Negate)
            # Negations are based on Actions in this specific shot
            current_actions = attrs.get('Actions', [])
            if 'Appearance' in attrs or current_actions:
                # Handle Negations
                negations = rules.get("attribute_negations", {})
                for action in current_actions:
                    if action in negations:
                        for negated_attr in negations[action]:
                            if negated_attr in persistence_tracker[slug]['Appearance']:
                                persistence_tracker[slug]['Appearance'].remove(negated_attr)
                
                # Merge new appearance attributes
                for app in attrs.get('Appearance', []):
                    if app not in persistence_tracker[slug]['Appearance']:
                        persistence_tracker[slug]['Appearance'].append(app)

            # Update Persistence State: Emotions (V12.1 Conflict Resolution)
            if 'Emotions' in attrs and attrs['Emotions']:
                current_emotions = list(persistence_tracker[slug]['Emotions'])
                new_emotions = attrs['Emotions']
                # Resolve conflicts, prioritizing the new emotions
                resolved_emotions = resolve_emotion_conflicts(current_emotions, new_emotions, rules.get("emotion_conflict_groups", []))
                persistence_tracker[slug]['Emotions'] = resolved_emotions


        # --- 4.3: Finalize Shot Composition (Applying Persistence & Presence) ---
        
        # V12.2: Guaranteed Scene Presence (Iterate over all present agents)
        for slug in scene_presence:
            # Determine if Character or Group
            is_group = slug in G_GROUPS_BY_SLUG
            comp_key = "Groups" if is_group else "Characters"

            is_active = slug in active_agents
            status = "active" if is_active else "passive"

            # Ensure agent is in composition
            shot_composition[comp_key][slug] = {"Status": status, "Attributes": {}}
            
            # Combine Attributes: Immediate (Shot) + Persistent
            
            # Start with Persistent (Appearance, Emotions)
            if slug in persistence_tracker:
                for attr_type in ["Appearance", "Emotions"]:
                    attrs = persistence_tracker[slug].get(attr_type, [])
                    if attrs:
                        shot_composition[comp_key][slug]["Attributes"][attr_type] = sorted(list(set(attrs)))

            # Add Immediate (Actions, and any immediate Appearance/Emotions not yet captured)
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


            # Apply Default Passive States (if passive and empty)
            if status == "passive":
                attrs = shot_composition[comp_key][slug]["Attributes"]
                if not attrs.get("Actions") and rules.get("default_passive_action"):
                    attrs.setdefault("Actions", []).append(rules["default_passive_action"])
                if not attrs.get("Emotions") and rules.get("default_passive_emotion"):
                     attrs.setdefault("Emotions", []).append(rules["default_passive_emotion"])

        
        # V12.4: Attribute Deduplication (Final Pass)
        # Characters
        for slug, data in shot_composition["Characters"].items():
            global_char_data = G_CHARS_BY_SLUG.get(slug)
            if global_char_data and 'Attributes' in data:
                data['Attributes'] = filter_inherent_attributes(data['Attributes'], global_char_data['inherent_attributes'])
                if not data['Attributes']:
                     del data['Attributes']
        
        # Groups
        for slug, data in shot_composition["Groups"].items():
             global_group_data = G_GROUPS_BY_SLUG.get(slug)
             if global_group_data and 'Attributes' in data:
                data['Attributes'] = filter_inherent_attributes(data['Attributes'], global_group_data.get('inherent_attributes', {}))
                if not data['Attributes']:
                     del data['Attributes']


        # Finalize Instructions (Deduplicate)
        for instr_type in shot_instructions:
            shot_instructions[instr_type] = sorted(list(set(shot_instructions[instr_type])))

        # Add finalized data to the shot
        shot["Shot_Composition"] = shot_composition
        shot["Shot_Type"] = determine_shot_type(shot_composition, shot_instructions)
        if shot_instructions:
            shot["Instructions"] = shot_instructions

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
# SECTION 6: MAIN EXECUTION (V12.7 Update)
# ===========================================================

def normalize_script(script_input:str, profile_name="PIXAR_3D_VI", title=None, is_file=True, output_dir="./output_normalized_v12"):
    
    # 1. Initialize Context (V12.7)
    try:
        ctx = initialize_context(profile_name)
    except ValueError as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return None

    # 2. Load Script Content
    if is_file:
        try:
            script_content = Path(script_input).read_text(encoding="utf-8")
            if title is None:
                title = Path(script_input).stem
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy file kịch bản tại {script_input}")
            return None
    else:
        script_content = script_input
        if title is None:
            title = "untitled_script"

    # 3. Pass 1: Global Character Consolidation
    try:
        global_map, global_characters, global_group_agents = global_character_pass(script_content, ctx)
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng trong quá trình Global Pass (Pass 1): {e}")
        return None

    # 4. Prepare Global Context for Scene Processing
    # Create lookups by Slug (for efficient access during scene processing)
    G_CHARS_BY_SLUG = {data['slug']: data for data in global_characters.values()}
    G_GROUPS_BY_SLUG = {data['slug']: data for data in global_group_agents.values()}

    # Create a unified lookup map (Case-insensitive term -> slug)
    agent_slug_lookup = {}
    for term, c_name in global_map.items():
        if c_name in global_characters:
            agent_slug_lookup[canonicalize(term).lower()] = global_characters[c_name]['slug']
    for g_name, data in global_group_agents.items():
        agent_slug_lookup[canonicalize(g_name).lower()] = data['slug']

    GLOBAL_CONTEXT = {
        'Global_Characters': global_characters,
        'Global_Group_Agents': global_group_agents,
        'Global_Characters_By_Slug': G_CHARS_BY_SLUG,
        'Global_Group_Agents_By_Slug': G_GROUPS_BY_SLUG,
        'Agent_Slug_Lookup': agent_slug_lookup
    }

    # 5. Scene Detection
    scenes_data = detect_scenes(script_content, ctx)
    
    # 6. Pass 2: Scene Processing and Shot Composition
    processed_scenes = []
    try:
        for sc_data in scenes_data:
            processed_scene = process_scene(sc_data, GLOBAL_CONTEXT, ctx)
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
        output_json_path = Path(output_dir,"storygrid_v12.json")
        
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(story, f, ensure_ascii=False, indent=2)
            print(f"✅ Hoàn tất (V12): Phân tích thành công {len(processed_scenes)} cảnh (Production-Ready).")
            print(f"🧾 Kết quả xuất tại: {output_json_path}")
        except Exception as e:
            print(f"❌ Lỗi khi ghi file JSON: {e}")

    return story

# ---------- CLI ----------
if __name__=="__main__":
    # Set stdout encoding to UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass # Handle environments where reconfigure is not available

    ap=argparse.ArgumentParser(description="Screenplay Normalizer (v12.0) → StoryGrid v4.0")
    ap.add_argument("--script", required=True, help="Đường dẫn file kịch bản .txt")
    ap.add_argument("--profile", default="PIXAR_3D_VI", help="Tên profile cấu hình (mặc định: PIXAR_3D_VI)")
    ap.add_argument("--output", default="./output_normalized_v12", help="Thư mục xuất kết quả")
    
    try:
        if len(sys.argv) > 1:
             args=ap.parse_args()
             normalize_script(args.script, profile_name=args.profile, output_dir=args.output, is_file=True)
        elif not 'ipykernel' in sys.modules:
             print("Chạy ở chế độ CLI yêu cầu tham số --script.")

    except SystemExit:
        pass