#!/usr/bin/python3

import requests
import json
from datetime import datetime, timedelta

# Define your Zabbix API URL, user credentials, and host group
URL = "https://ubersb-zabbix.lighthouse-cloud.com/api_jsonrpc.php"
USER = "api_u1"
PASS = "uVjaU5IJqzm@"
GROUP = "Uber_export"
# Function to call Zabbix API
def zabbix_api_call(method, params):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "auth": auth_token,
        "id": 1
    })
    response = requests.post(URL, headers=headers, data=payload)
    result = response.json()
    if 'error' in result:
        raise Exception("Zabbix API error: {}".format(result['error']['data']))
    return result

# Authenticate and get the auth token
try:
    auth_payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "user.login",
        "params": {
            "user": USER,
            "password": PASS
        },
        "id": 1
    })
    auth_response = requests.post(URL, headers={'Content-Type': 'application/json'}, data=auth_payload)
    auth_result = auth_response.json()
    if 'error' in auth_result:
        raise Exception("Zabbix API authentication error: {}".format(auth_result['error']['data']))
    auth_token = auth_result['result']
except Exception as e:
    print("Error during authentication: {}".format(e))
    exit(1)

try:
    # Get the group ID based on the group name
    group_params = {
        "output": "extend",
        "filter": {
            "name": [
                GROUP
            ]
        }
    }
    group_response = zabbix_api_call("hostgroup.get", group_params)
    if not group_response['result']:
        raise Exception("Host group '{}' not found.".format(GROUP))
    group_id = group_response['result'][0]['groupid']

    # Get hosts in the specified group
    hosts_params = {
        "output": ["hostid", "host"],
        "groupids": group_id
    }
    hosts_response = zabbix_api_call("host.get", hosts_params)
    if not hosts_response['result']:
        raise Exception("No hosts found in group '{}'.".format(GROUP))

    # Print the list of hosts and their items
    print("Hosts in group '{}':".format(GROUP))
    for host in hosts_response['result']:
        print("\nHost ID: {}, Host Name: {}".format(host['hostid'], host['host']))

        # Get items for the host
        items_params = {
            "output": ["itemid", "name", "lastvalue", "value_type", "state", "status"],
            "hostids": host['hostid'],
            "webitems": 1
        }
        items_response = zabbix_api_call("item.get", items_params)

        for item in items_response['result']:
            #skip unsupported items
            #print(item)
            if (item['state'] != '0') or (item['status'] != '0') :
                continue

            itemid = item['itemid']
            item_name = item['name']
            last_value = item['lastvalue']
            value_type = item['value_type']

            if host['host']=='uber_api_webscenario': #for webscenario hosts we want only response time items
                if not str(item_name).startswith('Response time'):
                    continue

            #Calculate the average value for the last hour
            now = datetime.now()
            time_from = int((now - timedelta(hours=1)).timestamp())
            time_till = int(now.timestamp())

            history_params = {
                "output": "extend",
                "history": value_type,
                "itemids": itemid,
                "time_from": time_from,
                "time_till": time_till
            }
            history_response = zabbix_api_call("history.get", history_params)

            if history_response['result']:
                values = [float(entry['value']) for entry in history_response['result']]
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
            else:
                avg_value = None
                continue

            print(f"  Item: {item_name}")
            print(f"    Last Value: {float(last_value):.2f}")
            print(f"    Last hour Values (avg min max): {avg_value:.2f} {min_value:.2f} {max_value:.2f}" )

except Exception as e:
    print("Error: {}".format(e))

finally:
    # Logout from Zabbix API
    try:
        zabbix_api_call("user.logout", {})
    except Exception as e:
        print("Error during logout: {}".format(e))
