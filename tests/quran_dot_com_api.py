import requests
import json


def get_recitations_from_quran_api(
        url="https://api.quran.com/api/v4/resources/recitations",
):
    """Get Reciters Metadata from api.quran.com
    Docs: https://api-docs.quran.com/docs/quran.com_versioned/recitations
    """

    payload = {}
    headers = {
        'Accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()


def get_recitation(
    recitation_id: int,
    base_url="https://api.quran.com/api/v4/quran/recitations",
):
    """Gets a recitation from api.quran.com
    """
    headers = {
        'Accept': 'application/json'
    }

    res = requests.get(f'{base_url}/{recitation_id}', headers=headers)
    return res.json()


if __name__ == '__main__':

    out = get_recitations_from_quran_api()
    print(json.dumps(out, indent=4))
    print(f'Len of recitations: {len(out['recitations'])}')
    print('\n' * 4)
    recitation = get_recitation(4)
    print(json.dumps(recitation, indent=4))

    """
    base url is: https://quran.app/
    """
