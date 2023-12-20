import gc
import logging
import os
import time
docs = "[Document(page_content='TBA Netting\n\nMBSD performs the TBA Netting process four times per month, " \
       "corresponding to each of the four primary settlement classes and dates established by Securities Industry " \
       "Financial Markets Association (“SIFMA”).12', metadata={'source': " \
       "'/home/llmuser/projects/plutooGPT/data/pdf/FICC_Disclosure_Framework.pdf'}), Document(page_content='3 TBA Net " \
       "Fully matched (and therefore novated) SBOD trades are processed through MBSD’s TBA netting service, " \
       "which is run according to the SIFMA calendar on 72hr day. This netting service supports SBOD trades submit in " \
       "GDM as well as odd-lots and will result in one or more SBON obligations versus FICC’s TBA novation account (" \
       "FTBA) priced at the TBA CUSIP’s “system price”8. A “TBA” cash differential will be generated to account for " \
       "the difference between SBOD trade price and system price.\n\nThe TBA Net has been expanded in the Novation " \
       "service to include odd-lot netting. In addition, all resulting obligations will be versus FICC’s TBA novation " \
       "account (FTBA) and therefore result in SBON obligations (SBO versus non-original counterparty); SBOO " \
       "obligations (SBO versus original counterparty) will no longer be generated.', metadata={'source': " \
       "'/home/llmuser/projects/plutooGPT/data/pdf/MBS_Novation_Messaging_Changes.pdf'}), Document(" \
       "page_content='Three business days prior to the SIFMA established settlement date (referred to as “72 Hour " \
       "Day”), TBA Netting for the applicable class occurs. On this date, all compared SBOD trades within the class " \
       "that has been designated for the TBA Netting process are netted with FICC as the counterparty. The net " \
       "positions created by the TBA Netting process are referred to as the settlement balance order position (“SBO " \
       "position”), which constitutes settlement obligations against which Members will (1) submit pool information (" \
       "“Pool Instructs”) for the Pool Netting process or (2) offset such SBO position with other SBO position or " \
       "trade-for-trade transaction, as applicable, through the DNA process. The Pool Netting process and the DNA " \
       "process are described in further detail below.', metadata={'source': " \
       "'/home/llmuser/projects/plutooGPT/data/pdf/FICC_Disclosure_Framework.pdf'}), Document(page_content='TBA (to " \
       "be announced) pass-throughs, or simply TBAs.\n\nMortgage options = options on TBAs.\n\nStructured MBSs such " \
       "as IOs, POs, CMOs: cash ﬂows are carved out from the cash ﬂows of the underlying pool of " \
       "collateral.\n\nConstant maturity mortgage (CMM) products.\n\n...\n\nA. Lesniewski\n\nInterest Rate and Credit " \
       "Models\n\nTBAs\n\nMBS Markets Modeling MBSs Prepayment and default modeling\n\nA TBA is a futures contract on " \
       "a pool of conventional, ﬁxed coupon mortgage loans.\n\nIt carries a coupon C reﬂective of the coupons on the " \
       "deliverable loans. The values of C are spaced in 50 bp increments: 3.5%, 4.0%, 4.5%, etc.\n\nThere is a " \
       "standard delivery date, the PSA date, in eachmonth. A vast majority of the trading takes place in either the " \
       "nearest or the once-deferred month, but the market quotes prices for three or four TBAs settling on the next " \
       "PSA dates.\n\nA party long a contract at settlement takes the delivery of a pool of mortgage loans satisfying " \
       "the good delivery guidelines.\n\nA. Lesniewski', metadata={'source': " \
       "'/home/llmuser/projects/plutooGPT/data/pdf/IRC_Lecture13_2019.pdf'})] "


def post_process_doc(sourceDocs):
    #print(sourceDocs)
    my_list = sourceDocs.split(",")
    # print(my_list)
    print("----------------------------------SOURCE DOCUMENTS---------------------------")
    for document in sourceDocs:
        #print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print("---------------------------")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    mydoc=post_process_doc(docs)
    # print(mydoc)
