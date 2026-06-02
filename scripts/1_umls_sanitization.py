python
import json
import logging
import re
import asyncio
import aiohttp
import os
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm

# --- SET UP LOGGING ---
logger = logging.getLogger("FLAIR_Audit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class AsyncUMLSClient:
    def __init__(self, api_key: str, max_concurrent_requests: int = 15):
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.cache = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    def _get_ngrams(self, text: str, max_n: int = 4) -> set:
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        ngrams = set()
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                ngrams.add(" ".join(words[i:i+n]))
        return ngrams

    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, params: dict, retries: int = 3):
        for attempt in range(retries):
            async with self.semaphore:
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if response.status != 200:
                            return None
                        return await response.json()
                except Exception:
                    await asyncio.sleep(1)
        return None

    async def check_term(self, session: aiohttp.ClientSession, term: str) -> bool:
        if len(term) < 4: return False
        if term in self.cache: return self.cache[term]

        search_url = f"{self.base_url}/search/current"
        search_data = await self._fetch_with_retry(session, search_url, {'string': term, 'searchType': 'exact', 'apiKey': self.api_key})

        if not search_data or search_data.get('result', {}).get('results', [])[0].get('ui') == 'NONE':
            self.cache[term] = False
            return False

        cui = search_data['result']['results'][0]['ui']
        cui_url = f"{self.base_url}/content/current/CUI/{cui}"
        cui_data = await self._fetch_with_retry(session, cui_url, {'apiKey': self.api_key})

        if not cui_data: return False

        for st in cui_data.get('result', {}).get('semanticTypes', []):
            uri = st.get('uri', '')
            if uri.endswith('T047') or uri.endswith('T046'):
                self.cache[term] = True
                return True

        self.cache[term] = False
        return False

    async def is_forbidden_clinical_term(self, session: aiohttp.ClientSession, text: str) -> bool:
        tasks = [self.check_term(session, term) for term in self._get_ngrams(text)]
        for f in asyncio.as_completed(tasks):
            if await f: return True
        return False

async def sanitize_prompts_async(input_filepath: str, output_filepath: str, api_key: str):
    logger.info("Initializing BioClinicalBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        
    umls_client = AsyncUMLSClient(api_key)
    sanitized_prompts = []
    
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(prompts, desc="Auditing Prompts")
        for item in pbar:
            clean_text = re.sub(r'\s*\\s*$', '', item.get("Prompt_Text", ""))
            family = item.get("Family", "")
            
            token_ids = tokenizer(clean_text)['input_ids']
            if len(token_ids) > 512 or (len(token_ids) > 0 and (token_ids.count(tokenizer.unk_token_id) / len(token_ids)) > 0.05):
                continue # Drop fragmented/long prompts
                
            if family in ["T_layman", "T_exclusion"]:
                if await umls_client.is_forbidden_clinical_term(session, clean_text):
                    continue # Drop forbidden terms

            item["Prompt_Text"] = clean_text
            sanitized_prompts.append(item)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(sanitized_prompts, f, indent=4)
    logger.info(f"Audit complete. Processed: {len(prompts)}. Surviving: {len(sanitized_prompts)}.")

if __name__ == "__main__":
    # IMPORTANT: Replace with your active NIH API Key, or load via environment variable.
    NIH_API_KEY = os.getenv("NIH_API_KEY", "YOUR_NIH_UMLS_API_KEY_HERE")
    if NIH_API_KEY == "YOUR_NIH_UMLS_API_KEY_HERE":
        logger.warning("No NIH API Key provided. Set the NIH_API_KEY environment variable or hardcode it for testing.")
    else:
        asyncio.run(sanitize_prompts_async("../data/raw_prompts.json", "../data/final_prompts.json", NIH_API_KEY))
