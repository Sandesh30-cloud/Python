import time

def inr_to_usd(val):
    return val / 83

def usd_to_inr(val):
    return val * 83

conversion_type = input("Type '1' for INR to USD conversion or '2' for USD to INR conversion: ")

val = float(input('Enter the amount: '))
time.sleep(3)
if conversion_type == '1':
    print(f'{val} INR to USD: {inr_to_usd(val):.2f} USD')
elif conversion_type == '2':
    print(f'{val} USD to INR: {usd_to_inr(val):.2f} INR')
else:
    print("Invalid input! Please enter '1' or '2' for the conversion type.")

