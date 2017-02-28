from pysb import Monomer, Parameter, Initial, Rule
from pysb.macros import bind_complex
from pysb.util import alias_model_components

def cox2_drugs_init(ibuprofen=False):

    if ibuprofen:
        Monomer('IBU', ['b']) #Ibuprofen
        Parameter('IBU_0', 180) #Micromolar
        alias_model_components()
        Initial(IBU(b=None), IBU_0)

def ibuprofen_binding_COX2(indp_cat=True, indp_allo=True):

    if indp_cat:
        #Rates for IBU binding at COX2 catalytic site (irrespective of allosteric site state)
        Parameter('kf_IBU_cat1', 1.5e4)
        Parameter('kr_IBU_cat1', 1.2e6)

        alias_model_components()
        bind_complex(COX2(cat=None), 'cat', IBU(b=None), 'b', [kf_IBU_cat1, kr_IBU_cat1])

    if indp_allo:
        #Rates for IBU binding at COX2 allosteric site (irrespective of catalytic site state)
        Parameter('kf_IBU_allo1', 1.5e4)
        Parameter('kr_IBU_allo1', 1.5e4)

        alias_model_components()

        Rule('bind_COX2allo_IBU',
             COX2(allo=None) + IBU(b=None) <> COX2(allo=1) % IBU(b=1),
             kf_IBU_allo1, kr_IBU_allo1)

def COX2_substrate_ibuprofen_binding():

    Rule('bind_COX2alloIBU_AA',
         COX2(allo=1, cat=None) % IBU(b=1) + AA(b=None) <> COX2(allo=1, cat=2) % IBU(b=1) % AA(b=2),
         kf_AA_cat1, kr_AA_cat1)

    Rule('bind_COX2alloIBU_AG',
         COX2(allo=1, cat=None) % IBU(b=1) + AG(b=None) <> COX2(allo=1, cat=2) % IBU(b=1) % AG(b=2),
         kf_AG_cat1, kr_AG_cat1)

    Rule('bind_COX2catIBU_AA',
         COX2(cat=1, allo=None) % IBU(b=1) + AA(b=None) <> COX2(allo=2, cat=1) % IBU(b=1) % AA(b=2),
         kf_AA_allo1, kr_AA_allo1)

    Rule('bind_COX2catIBU_AG',
         COX2(cat=1, allo=None) % IBU(b=1) + AG(b=None) <> COX2(allo=2, cat=1) % IBU(b=1) % AG(b=2),
         kf_AG_allo1, kr_AG_allo1)

def ibuprofen_COX2_catalysis(AAcat=True, AGcat=False):

    if AAcat:
        #This assumes the rate of catalysis with ibuprofen bound is the same as for the unbound enzyme
        Rule('kcat_AA_IBU',
             COX2(allo=1, cat=2) % IBU(b=1) % AA(b=2) >> COX2(allo=1, cat=None) % IBU(b=1) + PG(),
             kcat_AA1)

    if AGcat:
        pass

