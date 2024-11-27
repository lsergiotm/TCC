import asyncio
import asyncpg
import pandas as pd
# Dicionário de renomeação (mapeamento das colunas)
rename_mapping = {
    'qtd_0101': 'Quantidade de Ações coletivas/individuais em saúde',
    'qtd_0201': 'Quantidade das Coleta de material',
    'qtd_0202': 'Quantidade de Diagnósticos em laboratório clínico',
    'qtd_0203': 'Quantidade de Diagnósticos por anatomia patológica e citopatologia',
    'qtd_0204': 'Quantidade de Diagnósticos por radiologia',
    'qtd_0205': 'Quantidade de Diagnósticos por ultrasonografia',
    'qtd_0206': 'Quantidade de Diagnósticos por tomografia',
    'qtd_0207': 'Quantidade de Diagnósticos por ressonância magnética',
    'qtd_0208': 'Quantidade de Diagnósticos por medicina nuclear in vivo',
    'qtd_0209': 'Quantidade de Diagnósticos por endoscopia',
    'qtd_0210': 'Quantidade de Diagnósticos por radiologia intervencionista',
    'qtd_0211': 'Quantidade de Métodos diagnósticos em especialidades',
    'qtd_0212': 'Quantidade de Diagnósticos e procedimentos especiais em hemoterapia',
    'qtd_0214': 'Quantidade de Diagnósticos por teste rápido',
    'qtd_0301': 'Quantidade de Consultas / Atendimentos / Acompanhamentos',
    'qtd_0302': 'Quantidade de Fisioterapia',
    'qtd_0303': 'Quantidade de Tratamentos clínicos (outras especialidades)',
    'qtd_0304': 'Quantidade de Tratamentos em oncologia',
    'qtd_0305': 'Quantidade de Tratamentos em nefrologia',
    'qtd_0306': 'Quantidade de Hemoterapias',
    'qtd_0307': 'Quantidade de Tratamentos odontológicos',
    'qtd_0308': 'Quantidade de Tratamentos de lesões, envenenamentos e outros, decorrentes de causas externas',
    'qtd_0309': 'Quantidade de Terapias especializadas',
    'qtd_0310': 'Quantidade de Partos e nascimentos',
    'qtd_0401': 'Quantidade de Pequenas cirurgias e cirurgias de pele, tecido subcutâneo e mucosa',
    'qtd_0402': 'Quantidade de Cirurgias de glândulas endócrinas',
    'qtd_0403': 'Quantidade de Cirurgias do sistema nervoso central e periférico',
    'qtd_0404': 'Quantidade de Cirurgias das vias aéreas superiores, da face, da cabeça e do pescoço',
    'qtd_0405': 'Quantidade de Cirurgias do aparelho da visão',
    'qtd_0406': 'Quantidade de Cirurgias do aparelho circulatório',
    'qtd_0407': 'Quantidade de Cirurgias do aparelho digestivo, órgãos anexos e parede abdominal',
    'qtd_0408': 'Quantidade de Cirurgias do sistema osteomuscular',
    'qtd_0409': 'Quantidade de Cirurgias do aparelho geniturinário',
    'qtd_0410': 'Quantidade de Cirurgias de mama',
    'qtd_0411': 'Quantidade de Cirurgias obstétrica',
    'qtd_0412': 'Quantidade de Cirurgias torácica',
    'qtd_0413': 'Quantidade de Cirurgias reparadora',
    'qtd_0414': 'Quantidade de Bucomaxilofacial',
    'qtd_0415': 'Quantidade de Outras cirurgias',
    'qtd_0416': 'Quantidade de Cirurgias em oncologia',
    'qtd_0418': 'Quantidade de Cirurgias em nefrologia',
    'qtd_0501': 'Quantidade de Coletas e exames para fins de doação de órgãos, tecidos e células e de transplante',
    'qtd_0502': 'Quantidade de Avaliações de mortes encefálica',
    'qtd_0503': 'Quantidade de Ações relacionadas à doação de órgãos e tecidos para transplante',
    'qtd_0504': 'Quantidade de Processamentos de tecidos para transplante',
    'qtd_0505': 'Quantidade de Transplantes de órgãos, tecidos e células',
    'qtd_0506': 'Quantidade de Acompanhamentos e intercorrências no pré e pós-transplante',
    'qtd_0603': 'Quantidade de Medicamentos de âmbito hospitalar e urgência',
    'qtd_0702': 'Quantidade de Órteses, próteses e materiais especiais relacionados ao ato cirúrgico',
    'qtd_0801': 'Quantidade de Ações relacionadas ao estabelecimento',
    'qtd_0802': 'Quantidade de Ações relacionadas ao atendimento',
    'qtd_total': 'Quantidade total de procedimentos',
    'vl_0201': 'Valor das Coletas de material',
    'vl_0202': 'Valor dos Diagnósticos em laboratório clínico',
    'vl_0203': 'Valor dos Diagnósticos por anatomia patológica e citopatologia',
    'vl_0204': 'Valor dos Diagnósticos por radiologia',
    'vl_0205': 'Valor dos Diagnósticos por ultrasonografia',
    'vl_0206': 'Valor dos Diagnósticos por tomografia',
    'vl_0207': 'Valor dos Diagnósticos por ressonância magnética',
    'vl_0208': 'Valor dos Diagnósticos por medicina nuclear in vivo',
    'vl_0209': 'Valor dos Diagnósticos por endoscopia',
    'vl_0210': 'Valor dos Diagnósticos por radiologia intervencionista',
    'vl_0211': 'Valor dos Métodos diagnósticos em especialidades',
    'vl_0212': 'Valor dos Diagnósticos e procedimentos especiais em hemoterapia',
    'vl_0214': 'Valor dos Diagnósticos por teste rápido',
    'vl_02': 'Valor total dos procedimentos com finalidade diagnóstica',
    'vl_0301': 'Valor das Consultas / Atendimentos / Acompanhamentos',
    'vl_0302': 'Valor das Fisioterapias',
    'vl_0303': 'Valor dos Tratamentos clínicos (outras especialidades)',
    'vl_0304': 'Valor dos Tratamentos em oncologia',
    'vl_0305': 'Valor dos Tratamentos em nefrologia',
    'vl_0306': 'Valor das Hemoterapias',
    'vl_0307': 'Valor dos Tratamentos odontológicos',
    'vl_0308': 'Valor dos Tratamentos de lesões, envenenamentos e outros, decorrentes de causas externas',
    'vl_0309': 'Valor das Terapias especializadas',
    'vl_0310': 'Valor dos Partos e nascimento',
    'vl_03': 'Valor total dos procedimentos clínicos',
    'vl_0401': 'Valor das Pequenas cirurgias e cirurgias de pele, tecido subcutâneo e mucosa',
    'vl_0402': 'Valor das Cirurgias de glândulas endócrinas',
    'vl_0403': 'Valor das Cirurgias do sistema nervoso central e periférico',
    'vl_0404': 'Valor das Cirurgias das vias aéreas superiores, da face, da cabeça e do pescoço',
    'vl_0405': 'Valor das Cirurgias do aparelho da visão',
    'vl_0406': 'Valor das Cirurgias do aparelho circulatório',
    'vl_0407': 'Valor das Cirurgias do aparelho digestivo, órgãos anexos e parede abdominal',
    'vl_0408': 'Valor das Cirurgias do sistema osteomuscular',
    'vl_0409': 'Valor das Cirurgias do aparelho geniturinário',
    'vl_0410': 'Valor das Cirurgias de mama',
    'vl_0411': 'Valor das Cirurgias obstétrica',
    'vl_0412': 'Valor das Cirurgias torácica',
    'vl_0413': 'Valor das Cirurgias reparadora',
    'vl_0414': 'Valor das Bucomaxilofacial',
    'vl_0415': 'Valor das Outras cirurgias',
    'vl_0416': 'Valor das Cirurgias em oncologia',
    'vl_0417': 'Valor das Anestesiologia',
    'vl_0418': 'Valor das Cirurgias em nefrologia',
    'vl_04': 'Valor total dos procedimentos cirúrgicos',
    'vl_0501': 'Valor das Coletas e exames para fins de doação de órgãos, tecidos e células e de transplante',
    'vl_0502': 'Valor das Avaliações de mortes encefálica',
    'vl_0503': 'Valor das Ações relacionadas à doação de órgãos e tecidos para transplante',
    'vl_0504': 'Valor dos Processamentos de tecidos para transplante',
    'vl_0505': 'Valor das Transplantes de órgãos, tecidos e células',
    'vl_0506': 'Valor das Acompanhamentos e intercorrências no pré e pós-transplante',
    'vl_05': 'Valor total dos transplantes de órgãos, tecidos e células',
    'vl_0603': 'Valor dos Medicamentos de âmbito hospitalar e urgência',
    'vl_06': 'Valor total dos medicamentos',
    'vl_0702': 'Valor das Órteses, próteses e materiais especiais relacionados ao ato cirúrgico',
    'vl_07': 'Valor total das órteses, próteses e materiais especiais',
    'vl_0801': 'Valor das Ações relacionadas ao estabelecimento',
    'vl_0802': 'Valor das Ações relacionadas ao atendimento',
    'vl_08': 'Valor total das ações complementares da atenção à saúde',
    'vl_total': 'Valor total dos procedimentos',
    "QTD_0101": "Ações coletivas/individuais em saúde",
}

# Função para carregar os dados do banco de dados
async def fetch_data():
    conn = await asyncpg.connect(
        user='Data_IESB', 
        password='DATA_IESB', 
        database='Data_IESB', 
        host='dataiesb.iesbtech.com.br'
    )
    query = "SELECT * FROM saude_ride_tcc_luis"
    rows = await conn.fetch(query)
    await conn.close()
    
    # Convertendo para DataFrame
    df = pd.DataFrame(rows, columns=[col for col in rows[0].keys()])
    return df

# Função para carregar e processar os dados
def load_data():
    # Carregar dados assíncronos
    df = asyncio.run(fetch_data())

    # Substituir valores nulos por 0
    df = df.fillna(0)

    # Renomear as colunas do DataFrame
    df = df.rename(columns=rename_mapping)
    
    # Retornar o DataFrame processado
    return df

# Função para carregar o dicionário de renomeação
def load_rename_mapping():
    return rename_mapping