import pinecone
import openai
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import DocumentUploadForm
from .models import Document
from .utils import store_document_in_pinecone, search_similar_chunks
from pinecone import Pinecone, ServerlessSpec
import chardet

pc = Pinecone(api_key="pcsk_5UYoAU_KQQzpErejyp86WpXeQpNDKPvQJsKeyNroLNuCy5X4BLXYBkxTCZ9X16iAeYHwP4")

# Create or connect to your Pinecone index
index_name = "lesson-index"

# Check if the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=512,  # Use suitable dimension based on your embedding model
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Now you can interact with your index
index = pc.Index(index_name)

# Initialize OpenAI API
openai.api_key = settings.OPENAI_API_KEY

def upload_document(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)  # don't save yet
            uploaded_file = request.FILES['file']
            
            try:
                # Extract clean text from the file
                content = extract_text_from_file(uploaded_file)
                document.content = content
                document.save()  # Save after content is added

                # Send the clean content to Pinecone
                store_document_in_pinecone(document)

                return redirect('document_list')
            except Exception as e:
                return render(request, 'lesson_plans/upload_document.html', {
                    'form': form,
                    'error': f"Failed to process file: {str(e)}"
                })
    else:
        form = DocumentUploadForm()

    return render(request, 'lesson_plans/upload_document.html', {'form': form})



def semantic_search(query):
    from difflib import SequenceMatcher
    docs = Document.objects.all()
    results = []

    for doc in docs:
        score = SequenceMatcher(None, query.lower(), doc.text.lower()).ratio()
        if score > 0.3:  # threshold
            results.append({
                'text': doc.text,
                'score': score
            })

    return sorted(results, key=lambda x: x['score'], reverse=True)

def search_view(request):
    query = request.GET.get("query", "")
    results = []
    answer = None

    if query:
        try:
            result_data = search_similar_chunks(query, top_k=5, use_gpt=True)
            if isinstance(result_data, dict):
                answer = result_data.get("answer")
                results = result_data.get("chunks", [])
            else:
                results = result_data
        except Exception as e:
            return render(request, 'lesson_plans/search.html', {
                "query": query,
                "results": [],
                "error": f"Error during search: {str(e)}"
            })

    return render(request, "lesson_plans/search.html", {
        "query": query,
        "results": results,
        "answer": answer,
    })
