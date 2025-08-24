import os
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app, data_processor

client = TestClient(app)

def create_dummy_excel(path: str):
    df = pd.DataFrame({
        'S.No': [1, None],
        'Flight Number': ['AI123', None],
        'Date': [pd.Timestamp('2025-07-19'), pd.Timestamp('2025-07-20')],
        'From': ['Mumbai (BOM)', 'Mumbai (BOM)'],
        'To': ['Delhi (DEL)', 'Delhi (DEL)'],
        'Aircraft': ['A320', 'A320'],
        'Flight time': ['01:50', '01:50'],
        'STD': [pd.Timestamp('2025-07-19 09:00:00').time(), pd.Timestamp('2025-07-20 09:00:00').time()],
        'ATD': [pd.Timestamp('2025-07-19 09:05:00').time(), pd.Timestamp('2025-07-20 09:12:00').time()],
        'STA': [pd.Timestamp('2025-07-19 10:50:00').time(), pd.Timestamp('2025-07-20 10:50:00').time()],
        'ATA': ['Landed 10:58 AM', 'Landed 10:55 AM']
    })
    df.to_excel(path, index=False)


def test_upload_and_enrichment_flow(tmp_path):
    # Create dummy excel
    excel_path = tmp_path / 'dummy.xlsx'
    create_dummy_excel(excel_path)

    # Upload file
    with open(excel_path, 'rb') as f:
        response = client.post('/api/upload-data', files={'file': ('dummy.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['success'] is True
    assert data['processing_result']['records_processed'] >= 1

    # Fetch raw data
    r2 = client.get('/api/raw-data?limit=10')
    assert r2.status_code == 200
    raw = r2.json()
    assert raw['count'] >= 1

    # Enrichment endpoint (will be empty but should work)
    r3 = client.get('/api/enrichment?limit=10')
    assert r3.status_code == 200
    enrich = r3.json()
    assert 'records' in enrich

    # Cascade network (may have 0 if insufficient graph but should return keys)
    r4 = client.get('/api/cascade-network?limit=50')
    assert r4.status_code == 200
    net = r4.json()
    assert 'nodes' in net and 'edges' in net
