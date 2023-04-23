"""
Runs loadtesting against an ImageSearch service using a small set of queries.
"""

import time

from locust import HttpUser, task


class BasicLoadTest(HttpUser):
    @task
    def generate_batch(self):
        """Makes a simple generate query."""
        self.client.post(
            "/generate-batch",
            headers={"content-type": "application/json"},
            json={
                "prompt": [
                    "Generate MARS Banner in English:\nbusiness_name: Jon'''s cat cafe\nbusiness_vertical: Cafe\nbusiness_location: Los Angeles, CA",
                    "Generate MARS Testimonial in English:\nbusiness_name: Jon'''s cat cafe\nbusiness_vertical: Cafe\nbusiness_location: Los Angeles, CA",
                ],
            },
        )
