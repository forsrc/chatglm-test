import poplib
import email
from email.header import decode_header
from email.utils import parseaddr
from email import charset

email_address = "xxx"
password = "xxx"

pop_server = "pop.xxx"
port = 110

pop_conn = poplib.POP3(pop_server, port=port, timeout=10)

pop_conn.user(email_address)
pop_conn.pass_(password)

num_messages = len(pop_conn.list()[1])
print("Total emails in mailbox:", num_messages)


for i in range(num_messages):
    _, msg_data, _ = pop_conn.retr(i + 1)
    msg_text = b"\n".join(msg_data).decode("utf-8", errors="replace") 

    msg = email.message_from_string(msg_text)

    from_ = parseaddr(msg.get("From"))[1]
    subject, encoding = decode_header(msg.get("Subject"))[0]

    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8", errors="replace")

    print(f"From: {from_}")
    print(f"Subject: {subject}")

    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            body = part.get_payload(decode=True)

            charset_str = part.get_content_charset()
            if charset_str:
                charset_obj = charset.Charset(charset_str)
                body = body.decode(charset_obj.input_charset, errors="replace")
            else:
                body = body.decode("utf-8", errors="replace")

            print("Body:\n", body)

pop_conn.quit()
