import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv
import re
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
# from API_STORE import GROQ_API_KEY, PINECONE_API_KEY
from pinecone import Pinecone
# -------------------------------
# Configuration
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CHROMA_DB_PATH = "./chroma_db_2"
COLLECTION_NAME = "rada_chatbot_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq Model Options
GROQ_MODEL = "llama-3.3-70b-versatile"  # Options: "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TOP_K = 60
MAX_TOKENS = 6000

# -------------------------------
# Initialize Components
# -------------------------------
def init_pinecone():
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index_name = COLLECTION_NAME.lower().replace("_", "-")

    # Create index if it doesn't exist
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        from pinecone import ServerlessSpec
        # Determine embedding dimension
        from sentence_transformers import SentenceTransformer
        dimension = SentenceTransformer(EMBEDDING_MODEL).get_sentence_embedding_dimension()

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",       # or "gcp" depending on your Pinecone account
                region="us-east-1" # must match your Pinecone environment
            )
        )
        print(f"🆕 Created Pinecone index '{index_name}' (dim={dimension})")
    else:
        print(f"✅ Pinecone index '{index_name}' exists")

    # Connect to index
    index = pc.Index(index_name)
    print(f"✅ Connected to Pinecone index '{index_name}'")
    return index

def init_embedding_model():
    print(f"✅ Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def init_reranker():
    print(f"✅ Loading reranker model: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)

def init_groq():
    if not GROQ_API_KEY:
        raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables")
    client = Groq(api_key=GROQ_API_KEY)
    print(f"✅ Groq client initialized (Model: {GROQ_MODEL})")
    return client

class DataAnalyzer:
    """Advanced computational operations on production data"""
    
    @staticmethod
    def parse_production_value(value_str: str) -> float:
        """Extract numeric value from string"""
        try:
            cleaned = re.sub(r'[^\d\.\-]', '', str(value_str))
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    @staticmethod
    def extract_date(metadata: Dict) -> str:
        """Extract date from metadata"""
        for key in ['date', 'productionDate', 'Date', 'production_date', 'timestamp']:
            if key in metadata:
                date_val = str(metadata[key])
                return date_val.split('T')[0] if 'T' in date_val else date_val
        return None
    
    @staticmethod
    def is_date_in_range(date_str: str, start_date: str = None, end_date: str = None,
                        month: str = None, year: str = None) -> bool:
        """Check if date falls within range"""
        if not date_str:
            return False
        try:
            date = datetime.fromisoformat(date_str.split('T')[0])
            
            if year and str(date.year) != str(year):
                return False
            
            if month:
                month_num = datetime.strptime(month, "%B").month if month.isalpha() else int(month)
                if date.month != month_num:
                    return False
            
            if start_date and date < datetime.fromisoformat(start_date):
                return False
            if end_date and date > datetime.fromisoformat(end_date):
                return False
            return True
        except:
            return False
    
    @staticmethod
    def extract_production_from_text(text: str) -> Dict[str, float]:
        """Extract production values from text using multiple patterns"""
        production = {'oil': 0.0, 'gas': 0.0, 'water': 0.0, 'bsw': 0.0, 'gross': 0.0}
        
        patterns = {
            'oil': [
                r'(?:netOil|net_oil|oilProduction|oil.*?production).*?[:=]\s*([\d\.]+)',
                r'oil.*?[:=]\s*([\d\.]+)\s*bbl'
            ],
            'gas': [
                r'(?:netGas|net_gas|gasProduction|gas.*?production).*?[:=]\s*([\d\.]+)',
                r'gas.*?[:=]\s*([\d\.]+)\s*[Mm]scf'
            ],
            'water': [
                r'(?:netWater|net_water|waterProduction|water.*?production).*?[:=]\s*([\d\.]+)',
                r'water.*?[:=]\s*([\d\.]+)\s*bbl'
            ],
            'bsw': [r'(?:bsw|waterRF|water.*?cut).*?[:=]\s*([\d\.]+)'],
            'gross': [r'(?:gross|grossProduction).*?[:=]\s*([\d\.]+)']
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    production[key] = float(match.group(1))
                    break
        
        return production
    
    def aggregate_production(self, chunks: List[str], metadatas: List[Dict],
                            flowstation: str = None, well: str = None,
                            start_date: str = None, end_date: str = None,
                            month: str = None, year: str = None,
                            asset: str = None) -> Dict:
        """Comprehensive production aggregation with enhanced filtering"""
        
        results = {
            'total_oil': 0.0,
            'total_gas': 0.0,
            'total_water': 0.0,
            'total_gross': 0.0,
            'avg_bsw': 0.0,
            'bsw_count': 0,
            'record_count': 0,
            'wells': set(),
            'flowstations': set(),
            'dates': set(),
            'assets': set(),
            'records': []
        }
        
        for chunk, meta in zip(chunks, metadatas):
            # Apply filters
            if asset and not any(asset.lower() in str(v).lower() 
                               for k, v in meta.items() if 'asset' in k.lower()):
                continue
            
            if flowstation and not any(flowstation.lower() in str(v).lower()
                                      for k, v in meta.items() if 'flow' in k.lower() or 'station' in k.lower()):
                continue
            
            if well:
                well_found = False
                for k, v in meta.items():
                    if any(w in k.lower() for w in ['well', 'production', 'string']):
                        if well.lower() in str(v).lower():
                            well_found = True
                            break
                if not well_found:
                    continue
            
            # Date filtering
            record_date = self.extract_date(meta)
            if (start_date or end_date or month or year):
                if not self.is_date_in_range(record_date, start_date, end_date, month, year):
                    continue
            
            # Extract production from metadata
            oil = gas = water = bsw = gross = 0.0
            
            for key, value in meta.items():
                key_lower = key.lower()
                if ('oil' in key_lower and ('net' in key_lower or 'production' in key_lower)):
                    oil = max(oil, self.parse_production_value(value))
                elif ('gas' in key_lower and ('net' in key_lower or 'production' in key_lower)):
                    gas = max(gas, self.parse_production_value(value))
                elif ('water' in key_lower and ('net' in key_lower or 'production' in key_lower)):
                    water = max(water, self.parse_production_value(value))
                elif 'gross' in key_lower:
                    gross = max(gross, self.parse_production_value(value))
                elif 'bsw' in key_lower or 'waterrf' in key_lower:
                    bsw = max(bsw, self.parse_production_value(value))
            
            # Fallback to text extraction
            if oil == 0.0 and gas == 0.0 and water == 0.0:
                text_production = self.extract_production_from_text(chunk)
                oil = text_production['oil']
                gas = text_production['gas']
                water = text_production['water']
                if bsw == 0.0:
                    bsw = text_production['bsw']
                if gross == 0.0:
                    gross = text_production['gross']
            
            # Skip if no meaningful data
            if oil == 0.0 and gas == 0.0 and water == 0.0 and gross == 0.0:
                continue
            
            # Aggregate
            results['total_oil'] += oil
            results['total_gas'] += gas
            results['total_water'] += water
            results['total_gross'] += gross if gross > 0 else (oil + water)
            
            if bsw > 0:
                results['avg_bsw'] += bsw
                results['bsw_count'] += 1
            
            # Track metadata
            for key, value in meta.items():
                if any(w in key.lower() for w in ['well', 'production', 'string', 'listid']):
                    if value:
                        results['wells'].add(str(value))
                if any(w in key.lower() for w in ['flowstation', 'flow_station']):
                    if value:
                        results['flowstations'].add(str(value))
                if 'asset' in key.lower():
                    if value:
                        results['assets'].add(str(value))
            
            if record_date:
                results['dates'].add(record_date)
            
            results['record_count'] += 1
            results['records'].append({
                'oil': round(oil, 2),
                'gas': round(gas, 4),
                'water': round(water, 2),
                'gross': round(gross, 2),
                'bsw': round(bsw, 4),
                'date': record_date
            })
        
        # Calculate averages
        if results['bsw_count'] > 0:
            results['avg_bsw'] = round(results['avg_bsw'] / results['bsw_count'], 4)
        
        # Convert sets to sorted lists
        results['wells'] = sorted(list(results['wells']))
        results['flowstations'] = sorted(list(results['flowstations']))
        results['assets'] = sorted(list(results['assets']))
        results['dates'] = sorted(list(results['dates']))
        
        # Round totals
        results['total_oil'] = round(results['total_oil'], 2)
        results['total_gas'] = round(results['total_gas'], 4)
        results['total_water'] = round(results['total_water'], 2)
        results['total_gross'] = round(results['total_gross'], 2)
        
        return results
    
    def find_top_producers(self, chunks: List[str], metadatas: List[Dict],
                          flowstation: str = None, asset: str = None,
                          metric: str = 'oil', limit: int = 5,
                          date_filter: str = None) -> List[Dict]:
        """Find top producers by metric"""
        well_production = defaultdict(lambda: {
            'oil': 0.0, 'gas': 0.0, 'water': 0.0, 'gross': 0.0,
            'flowstation': '', 'asset': '', 'dates': set()
        })
        
        for chunk, meta in zip(chunks, metadatas):
            # Apply filters
            if flowstation and not any(flowstation.lower() in str(v).lower()
                                      for k, v in meta.items() if 'flow' in k.lower()):
                continue
            
            if asset and not any(asset.lower() in str(v).lower()
                               for k, v in meta.items() if 'asset' in k.lower()):
                continue
            
            if date_filter:
                record_date = self.extract_date(meta)
                if not record_date or date_filter not in record_date:
                    continue
            
            # Extract well identifier
            well = None
            for key, value in meta.items():
                if any(w in key.lower() for w in ['well', 'production', 'string', 'listid']):
                    if value:
                        well = str(value)
                        break
            
            if not well:
                continue
            
            # Extract production
            oil = gas = water = gross = 0.0
            for key, value in meta.items():
                key_lower = key.lower()
                if 'oil' in key_lower and ('net' in key_lower or 'production' in key_lower):
                    oil += self.parse_production_value(value)
                elif 'gas' in key_lower and ('net' in key_lower or 'production' in key_lower):
                    gas += self.parse_production_value(value)
                elif 'water' in key_lower and ('net' in key_lower or 'production' in key_lower):
                    water += self.parse_production_value(value)
                elif 'gross' in key_lower:
                    gross += self.parse_production_value(value)
            
            well_production[well]['oil'] += oil
            well_production[well]['gas'] += gas
            well_production[well]['water'] += water
            well_production[well]['gross'] += gross if gross > 0 else (oil + water)
            
            # Store metadata
            for k, v in meta.items():
                if 'flowstation' in k.lower() or 'flow_station' in k.lower():
                    well_production[well]['flowstation'] = str(v)
                if 'asset' in k.lower():
                    well_production[well]['asset'] = str(v)
            
            record_date = self.extract_date(meta)
            if record_date:
                well_production[well]['dates'].add(record_date)
        
        # Sort by metric
        sorted_wells = sorted(
            well_production.items(),
            key=lambda x: x[1][metric],
            reverse=True
        )[:limit]
        
        return [{
            'well': well,
            'oil': round(data['oil'], 2),
            'gas': round(data['gas'], 4),
            'water': round(data['water'], 2),
            'gross': round(data['gross'], 2),
            'flowstation': data['flowstation'],
            'asset': data['asset'],
            'dates': sorted(list(data['dates']))
        } for well, data in sorted_wells]
    
    def count_wells(self, chunks: List[str], metadatas: List[Dict],
                   flowstation: str = None, asset: str = None,
                   date: str = None, month: str = None, year: str = None) -> Dict:
        """Count unique wells with comprehensive filters"""
        wells = set()
        well_details = []
        
        for chunk, meta in zip(chunks, metadatas):
            if flowstation and not any(flowstation.lower() in str(v).lower()
                                      for k, v in meta.items() if 'flow' in k.lower()):
                continue
            
            if asset and not any(asset.lower() in str(v).lower()
                               for k, v in meta.items() if 'asset' in k.lower()):
                continue
            
            record_date = self.extract_date(meta)
            if date and (not record_date or date not in record_date):
                continue
            if (month or year) and not self.is_date_in_range(record_date, None, None, month, year):
                continue
            
            # Extract well
            well = None
            for key, value in meta.items():
                if any(w in key.lower() for w in ['well', 'production', 'string', 'listid']):
                    if value:
                        well = str(value)
                        break
            
            if well:
                wells.add(well)
                well_details.append({
                    'well': well,
                    'flowstation': next((str(v) for k, v in meta.items() if 'flow' in k.lower()), 'Unknown'),
                    'asset': next((str(v) for k, v in meta.items() if 'asset' in k.lower()), 'Unknown'),
                    'date': record_date
                })
        
        return {
            'count': len(wells),
            'wells': sorted(list(wells)),
            'details': well_details
        }

# -------------------------------
# Query Classification
# -------------------------------
def classify_query(query: str) -> Dict[str, Any]:
    """Smart query classification with intent detection"""
    query_lower = query.lower()
    
    classification = {
        'needs_computation': False,
        'operation': None,
        'filters': {},
        'intent': 'lookup'
    }
    
    # Intent detection
    if any(w in query_lower for w in ['compare', 'versus', 'vs', 'difference']):
        classification['intent'] = 'compare'
        classification['needs_computation'] = True
        classification['operation'] = 'compare'
    elif any(w in query_lower for w in ['total', 'sum', 'cumulative', 'average', 'mean']):
        classification['intent'] = 'aggregate'
        classification['needs_computation'] = True
    elif any(w in query_lower for w in ['list', 'all', 'show me', 'give me']):
        classification['intent'] = 'list'
    
    # Operation keywords
    computation_keywords = {
        'sum': ['total', 'sum', 'cumulative'],
        'average': ['average', 'mean', 'avg'],
        'count': ['how many', 'number of', 'count'],
        'max': ['highest', 'maximum', 'max', 'top', 'best'],
        'min': ['lowest', 'minimum', 'min', 'bottom', 'worst']
    }
    
    for operation, keywords in computation_keywords.items():
        if any(kw in query_lower for kw in keywords):
            classification['needs_computation'] = True
            if not classification['operation']:
                classification['operation'] = operation
            break
    
    # Extract filters
    fs_patterns = [
        r'(awoba|ekulama\s*[12]?|efe|obi\s*anyima)\s*(?:flowstation|fs)?',
        r'flowstation\s+(\w+)'
    ]
    for pattern in fs_patterns:
        match = re.search(pattern, query_lower)
        if match:
            classification['filters']['flowstation'] = match.group(1).strip()
            break
    
    # Asset
    if 'oml' in query_lower:
        oml_match = re.search(r'oml\s*(\d+)', query_lower)
        if oml_match:
            classification['filters']['asset'] = f"OML {oml_match.group(1)}"
    
    # Well
    well_match = re.search(r'([A-Z]{4}\d{3}[A-Z]?:[A-Z]\d{3,4}[A-Z]?)', query.upper())
    if well_match:
        classification['filters']['well'] = well_match.group(1)
    
    # Date/Month/Year
    month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    for month in month_names:
        if month in query_lower:
            classification['filters']['month'] = month.capitalize()
            break
    
    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        classification['filters']['year'] = year_match.group(1)
    
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
    if date_match:
        classification['filters']['date'] = date_match.group(1)
    
    return classification

# -------------------------------
# Main Query Function
# -------------------------------
def query_chatbot_with_compute(collection, embedding_model, reranker, openai_client,
                               user_question: str, debug=False) -> str:
    
    classification = classify_query(user_question)
    
    if debug:
        print(f"\n🔍 Classification: {json.dumps(classification, indent=2)}")
    
    # Query expansion
    queries = [user_question]
    if classification['filters'].get('flowstation'):
        queries.append(f"flowstation {classification['filters']['flowstation']}")
    if classification['filters'].get('well'):
        queries.append(classification['filters']['well'])
    
    # Retrieve
    all_chunks = []
    all_metadatas = []
    
    for query in queries[:3]:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True
        )

        for match in results.matches:
            if hasattr(match, "metadata") and "text" in match.metadata:
                all_chunks.append(match.metadata["text"])
                all_metadatas.append(match.metadata)

    
    if not all_chunks:
        return "❌ No relevant information found in the database."
    
    # Deduplicate
    seen = set()
    unique_chunks = []
    unique_metadatas = []
    for chunk, meta in zip(all_chunks, all_metadatas):
        chunk_id = chunk[:100]
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)
    
    # Rerank
    pairs = [(user_question, doc) for doc in unique_chunks]
    scores = reranker.predict(pairs)
    reranked_with_scores = sorted(
        zip(unique_chunks, unique_metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )
    
    reranked_chunks = [chunk for chunk, _, _ in reranked_with_scores]
    reranked_metadatas = [meta for _, meta, _ in reranked_with_scores]
    
    # Compute
    computed_result = None
    analyzer = DataAnalyzer()
    filters = classification['filters']
    
    if classification['needs_computation']:
        if classification['operation'] in ['sum', 'average']:
            computed_result = analyzer.aggregate_production(
                reranked_chunks, reranked_metadatas,
                flowstation=filters.get('flowstation'),
                well=filters.get('well'),
                month=filters.get('month'),
                year=filters.get('year'),
                asset=filters.get('asset')
            )
        elif classification['operation'] == 'count':
            computed_result = analyzer.count_wells(
                reranked_chunks, reranked_metadatas,
                flowstation=filters.get('flowstation'),
                asset=filters.get('asset'),
                month=filters.get('month'),
                year=filters.get('year')
            )
        elif classification['operation'] in ['max', 'min']:
            metric = 'oil'
            if 'gas' in user_question.lower():
                metric = 'gas'
            elif 'water' in user_question.lower():
                metric = 'water'
            
            computed_result = analyzer.find_top_producers(
                reranked_chunks, reranked_metadatas,
                flowstation=filters.get('flowstation'),
                asset=filters.get('asset'),
                metric=metric,
                limit=5
            )
        elif classification['intent'] == 'compare':
            # Extract both flowstations/assets to compare
            entities = re.findall(r'(awoba|ekulama\s*[12]?|oml\s*\d+)', user_question.lower())
            if len(entities) >= 2:
                results_compare = []
                for entity in entities[:2]:
                    fs = entity if 'oml' not in entity else None
                    asset = entity if 'oml' in entity else None
                    result = analyzer.aggregate_production(
                        reranked_chunks, reranked_metadatas,
                        flowstation=fs, asset=asset
                    )
                    result['entity'] = entity
                    results_compare.append(result)
                computed_result = {'comparison': results_compare}
    
    # Build context
    context_parts = []
    for chunk, meta in zip(reranked_chunks[:25], reranked_metadatas[:25]):
        summary = []
        if 'collection' in meta:
            summary.append(f"[{meta['collection']}]")
        for key in ['asset', 'assetName', 'flowStation', 'flowstation']:
            if key in meta:
                summary.append(f"[{meta[key]}]")
                break
        context_parts.append(f"{' '.join(summary)}\n{chunk[:600]}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Build prompt
    computation_text = ""
    if computed_result:
        computation_text = f"\n\n**COMPUTED ANALYSIS:**\n```json\n{json.dumps(computed_result, indent=2, default=str)}\n```\n"
    
    prompt = f"""You are a petroleum engineering data analyst providing accurate, well-formatted answers.

**INSTRUCTIONS:**
1. Use the COMPUTED ANALYSIS data - it contains accurate calculations
2. Present data in clean markdown tables when showing multiple items
3. Include exact values with units (bbl for barrels, Mscf for gas)
4. For comparisons, highlight key differences
5. Be concise but complete
6. If asking about specific entities (wells, flowstations), list them clearly

**CONTEXT:**
{context_text}
{computation_text}

**QUESTION:**
{user_question}

**ANSWER (be direct and well-formatted):**"""
    
    return groq_llm_inference(openai_client, prompt)

# -------------------------------
# [COPY ALL DataAnalyzer class methods from OpenAI version]
# [COPY classify_query function]
# [COPY query_chatbot_with_compute function - but use groq_llm_inference]
# -------------------------------

# I'll provide the key differences:

def groq_llm_inference(groq_client, prompt: str) -> str:
    """Call Groq API with Llama model"""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a petroleum engineering analyst. Provide accurate, well-formatted answers with tables where appropriate. Be direct and comprehensive."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
            top_p=0.9,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API Error: {e}"

# NOTE: Copy the ENTIRE DataAnalyzer class and query_chatbot_with_compute function
# from the OpenAI version - they're identical, just replace openai_llm_inference 
# with groq_llm_inference in the query_chatbot_with_compute function

def chatbot(debug=False):
    collection = init_pinecone()
    embedding_model = init_embedding_model()
    reranker = init_reranker()
    groq_client = init_groq()

    print("\n" + "="*80)
    print(f"🧠 RADA Petroleum Engineering Assistant (Groq {GROQ_MODEL})")
    print("="*80)
    print("I can analyze production data, compare flowstations, and perform calculations.")
    print("Type 'exit' to quit | Type 'debug' for diagnostics\n")

    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        
        if question.lower() == "debug":
            debug = not debug
            print(f"🔧 Debug: {'ON' if debug else 'OFF'}")
            continue
        
        if not question:
            continue
        
        try:
            print("\n⏳ Analyzing...")
            # NOTE: Pass groq_client instead of openai_client
            answer = query_chatbot_with_compute(
                collection, embedding_model, reranker,
                groq_client, question, debug=debug
            )
            print("\n" + "="*80)
            print("📊 ANSWER:")
            print("="*80)
            print(answer)
            print("="*80)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            if debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    chatbot(debug=False)