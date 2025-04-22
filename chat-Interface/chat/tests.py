from django.test import TestCase, Client
from django.urls import reverse
from .models import Chunk

class SaveVectorizationTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('save_vectorization')
        self.data = [
            {
                "document_id": 1,
                "content": "This is a test content",
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "document_id": 2,
                "content": "Another test content",
                "embedding": [0.4, 0.5, 0.6]
            }
        ]

    def test_save_vectorization(self):
        response = self.client.post(self.url, data=json.dumps(self.data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Chunk.objects.count(), 2)
        self.assertEqual(Chunk.objects.first().content, "This is a test content")

