#Creates a model with COX2, AA, 2-AG, and ibuprofen with the following assumptions:
#Ibuprofen can bind at catalytic or allosteric sites on COX2
#Its binding rate does not depend on whether anything else is bound to COX2
#AA catalysis is unaffected by ibuprofen in allosteric site
#2-AG catalysis is completely inhibited by ibuprofen in allosteric site

from corm import model
from drug_module import cox2_drugs_init, ibuprofen_binding_COX2, ibuprofen_COX2_catalysis

cox2_drugs_init(ibuprofen=True)
ibuprofen_binding_COX2(indp_cat=True, indp_allo=True)
ibuprofen_COX2_catalysis(AAcat=True, AGcat=False)