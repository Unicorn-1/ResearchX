from django import forms
from .widgets import MultipleFileInput

class DocumentForm(forms.Form):
    pdf_files = forms.FileField(widget=MultipleFileInput, required=True)
    question = forms.CharField(max_length=255, required=False)
