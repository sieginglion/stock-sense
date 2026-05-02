import asyncio
import datetime
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Literal, NamedTuple

import arrow
import dash_bootstrap_components as dbc
import dotenv
import numpy as np
import pandas as pd
import requests as rq
from dash import Dash, Input, Output, State, callback, dcc, html
from httpx import AsyncClient
from plotly import graph_objects as go
from plotly.subplots import make_subplots

dotenv.load_dotenv()

FMP_KEY = os.environ['FMP_KEY']

CACHE = "ON_TWSE.pkl"
URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
PATCH_DIR = Path('patch')
MARKET_TO_TIMEZONE = {
    'j': 'Asia/Tokyo',
    't': 'Asia/Taipei',
    'u': 'America/New_York',
}

if os.path.isfile(CACHE):
    with open(CACHE, "rb") as f:
        ON_TWSE = pickle.load(f)
else:
    df = pd.read_html(rq.get(URL, verify=False).text)[0]
    ON_TWSE = {
        r.iloc[0].split("\u3000", 1)[0]
        for _, r in df.iterrows()
        if r.iloc[5] == "ESVUFR"
    }
    with open(CACHE, "wb") as f:
        pickle.dump(ON_TWSE, f)


def add_suffix(market: Literal['j', 't', 'u'], symbol: str):
    return (
        symbol
        + {
            'j': '.T',
            't': '.TW' if symbol in ON_TWSE else '.TWO',
            'u': '',
        }[market]
    )


USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'

MARGIN = '50px'
FONT_COLOR = '#7b8ab8'
FONT = dict(
    color=FONT_COLOR,
    family='Nunito,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol"',
    size=14,
)

BAND_COLORS = [
    "#FF0000",
    "#FF0000",
    "#FFBF00",
    "#80FF00",
    "#00FF40",
    "#00FFFF",
    "#0040FF",
    "#8000FF",
    "#FF00BF",
]
BLUE = '#8eacd5'
DARK_GREEN = '#acd58e'
DARK_RED = '#d58eac'
LIGHT_GREEN = '#c8e3b4'
LIGHT_RED = '#e3b4c8'
TRANSPARENT = 'rgba(0, 0, 0, 0)'

app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id='market',
                    options=[
                        {'label': 'j', 'value': 'j'},
                        {'label': 't', 'value': 't'},
                        {'label': 'u', 'value': 'u'},
                    ],
                    value='t',
                    clearable=False,
                    style=dict(width='70px', marginRight=MARGIN),
                ),
                dbc.Input(
                    'input',
                    dict(textAlign='center', width='140px', marginRight=MARGIN),
                    placeholder='TSLA, 2330',
                    value='3131',
                ),
                dbc.Input(
                    'q',
                    dict(textAlign='center', width='70px', marginRight=MARGIN),
                    value=4,
                    type='number',
                ),
                dcc.Checklist(
                    id='ema7',
                    options=[{'label': 'EMA7', 'value': 'on'}],
                    value=[],
                ),
                dbc.Button('Plot', 'button'),
            ],
            style=dict(display='flex', marginTop=MARGIN),
        ),
        html.Div(
            [
                dcc.Markdown('', id='fmp-url'),
            ],
            style=dict(
                alignItems='center',
                display='flex',
                marginTop='16px',
                maxWidth='800px',
            ),
        ),
        dcc.Graph(
            'graph',
            config=dict(displayModeBar=False),
            figure=dict(data=[go.Sankey()], layout=dict(paper_bgcolor=TRANSPARENT)),
            style=dict(
                borderRadius='50px',
                boxShadow='5px 5px 10px rgba(55, 94, 148, 0.2), -5px -5px 10px rgba(255, 255, 255, 0.4)',
                height='1300px',
                marginTop=MARGIN,
                width='800px',
            ),
        ),
        dcc.ConfirmDialog('alert', 'Not Supported', displayed=False),
    ],
    style=dict(
        alignItems='center',
        display='flex',
        flexDirection='column',
        paddingBottom=MARGIN,
    ),
)


class NotSupported(Exception): ...


def get_item_from_sec(cik: str, tag: str, filing_dates: pd.Series):
    cmd = f'''
    curl 'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json' \
    -H 'accept-language: en-US,en;q=0.9' \
    -H 'cache-control: no-cache' \
    -H 'user-agent: {USER_AGENT}' \
    --compressed
    '''
    res = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        raise NotSupported
    df = pd.DataFrame(data['units']['USD'])
    df = df[df['form'].isin(['10-Q', '10-K']) & df['frame'].notna()].reset_index(
        drop=True
    )
    for i in df[df['form'] == '10-K'].index:
        if ((q := df.loc[i - 3 : i - 1])['form'] == '10-Q').sum() == 3:
            df.at[i, 'val'] -= q['val'].sum()
        else:
            df = df.drop(i)

    def find_filing_date(end_date: datetime.date):
        result = filing_dates[
            (end_date < filing_dates)
            & (filing_dates < end_date + pd.Timedelta(days=91))
        ]
        return result.iloc[0] if len(result) else pd.NA

    df['filing_date'] = pd.to_datetime(df['end']).dt.date.map(find_filing_date)
    df = df.set_index('filing_date')
    try:
        s = df.loc[filing_dates, 'val']
    except KeyError:
        raise NotSupported
    return s.reset_index(drop=True)


class Income(NamedTuple):
    d: datetime.date
    r: int
    cor: int
    gp: int
    oe: int
    oi: int
    rnd: int
    sgna: int
    eps: int
    rps: int


async def fetch_income_statements(
    market: Literal['j', 't', 'u'], symbol: str, limit: int
) -> pd.DataFrame:
    params = {
        'apikey': FMP_KEY,
        'limit': limit,
        'period': 'quarter',
    }
    if market == 'u':
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
    else:
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ add_suffix(market, symbol) }'
    async with AsyncClient() as client:
        data = (await client.get(url, params=params)).json()
    path = PATCH_DIR / f'{symbol}.json'
    if path.exists():
        with path.open() as f:
            patch = json.load(f)
        data = sorted(
            {r['date']: r for r in data + patch}.values(),
            key=lambda r: r['date'],
            reverse=True,
        )[:limit]
    if len(data) != limit:
        raise ValueError
    return pd.DataFrame(data).sort_values('date').reset_index(drop=True)


def get_incomes_from_fmp(market: Literal['j', 't', 'u'], symbol: str, q: int):
    df = asyncio.run(fetch_income_statements(market, symbol, q + 4))
    eps_col = 'epsDiluted' if market == 'u' else 'epsdiluted'

    def get_series(col_name):
        return df.get(col_name, pd.Series(0, df.index))

    shares = get_series('weightedAverageShsOutDil')
    df['eps_ttm'] = get_series(eps_col).rolling(4).sum()
    df['rps_ttm'] = (get_series('revenue') / shares).rolling(4).sum()
    df = df.iloc[3:]

    d = (pd.to_datetime(df['date']) + pd.Timedelta(days=1)).dt.date
    r = get_series('revenue')
    gp = get_series('grossProfit')
    oi = get_series('operatingIncome')
    rnd = get_series('researchAndDevelopmentExpenses')
    sgna = get_series('sellingGeneralAndAdministrativeExpenses')
    eps_ttm = get_series('eps_ttm')
    rps_ttm = get_series('rps_ttm')

    return [
        Income(*_)
        for _ in zip(d, r, r - gp, gp, gp - oi, oi, rnd, sgna, eps_ttm, rps_ttm)
    ]


# def get_incomes_from_finmind(symbol: str, q: int):
#     api = DataLoader()
#     api.login_by_token(FINMIND_KEY)
#     df = api.taiwan_stock_financial_statement(
#         stock_id=symbol,
#         start_date=arrow.now('Asia/Taipei')
#         .shift(days=-((q + 5) * 91))
#         .format('YYYY-MM-DD'),
#     )
#     # Pivot to wide format
#     df = df.pivot(index='date', columns='type', values='value').reset_index()
#     # We need at least Q + 4 records for rolling calcs
#     if len(df) < 4:
#         raise NotSupported
#     # Sort by date ascending to ensure calculations like rolling work correctly (Oldest -> Newest)
#     df = df.sort_values('date').reset_index(drop=True)

#     def get_series(col_name):
#         return df.get(col_name, pd.Series([0] * len(df)))

#     # 1. Calc rolling metrics (TTM) using full history
#     r_raw = get_series('Revenue')
#     eps_raw = get_series('EPS')
#     net_income = df.get('EquityAttributableToOwnersOfParent', df['IncomeAfterTaxes'])

#     shares = net_income / eps_raw
#     df['eps_ttm'] = eps_raw.rolling(4).sum()
#     df['rps_ttm'] = (r_raw / shares).rolling(4).sum()

#     # 2. Slice to remove the first 3 quarters (used for rolling warm-up)
#     df = df.iloc[3:].reset_index(drop=True)

#     # 3. Get quarterly series for the remaining valid range
#     d = (pd.to_datetime(df['date']) + pd.Timedelta(days=1)).dt.date
#     r = get_series('Revenue')
#     gp = get_series('GrossProfit')
#     oi = get_series('OperatingIncome')
#     rnd = get_series(None)
#     sgna = get_series(None)
#     eps_ttm = get_series('eps_ttm')
#     rps_ttm = get_series('rps_ttm')

#     return [
#         Income(*_)
#         for _ in zip(d, r, r - gp, gp, gp - oi, oi, rnd, sgna, eps_ttm, rps_ttm)
#     ]


# def get_incomes_from_tokenterminal(symbol: str):
#     slug = SLUG_TABLE[symbol]
#     url = 'https://api.tokenterminal.com/trpc/projects.getFinancialStatement'
#     params = {
#         'batch': '1',
#         'input': json.dumps({'0': {'project_slug': slug, 'granularity': 'month'}}),
#     }
#     headers = {
#         'accept': '*/*',
#         'accept-language': 'en-US,en;q=0.9',
#         'authorization': 'Bearer c0e5035a-64f6-4d2c-b5f6-ac1d1cb3da2f',
#         'cache-control': 'no-cache',
#         'content-type': 'application/json',
#         'origin': 'https://tokenterminal.com',
#         'pragma': 'no-cache',
#         'priority': 'u=1, i',
#         'referer': f'https://tokenterminal.com/explorer/projects/{slug}/financial-statement',
#         'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
#         'sec-ch-ua-mobile': '?0',
#         'sec-ch-ua-platform': '"macOS"',
#         'sec-fetch-dest': 'empty',
#         'sec-fetch-mode': 'cors',
#         'sec-fetch-site': 'same-site',
#         'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
#         'x-app-path': f'/explorer/projects/{slug}/financial-statement',
#         'x-tt-terminal-jwt': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcm9udEVuZCI6InRlcm1pbmFsIGRhc2hib2FyZCIsImlhdCI6MTc2NjUzNjU1MCwiZXhwIjoxNzY3NzQ2MTUwfQ.OzHHP4v66yYrUMoNrcQwU9rcausdKce4zQgzvjZnhIw',
#         'Cookie': '_ga=GA1.1.46309005.1766620105; _fbp=fb.1.1766620105447.33106851716235377; _gcl_au=1.1.2143769807.1766620106; intercom-id-p3bihfmm=f7640d7e-5b8d-4587-b06c-aa6f1c339bac; intercom-session-p3bihfmm=; intercom-device-id-p3bihfmm=dc4b28d8-a76b-4f46-ab88-0c17c25b10ba; _ga_TJ9TEYJ3GF=GS2.1.s1766623564$o2$g0$t1766623577$j47$l0$h0; ph_phc_amGyrGA1TpwJYYk2zNff9qfQkFBzu4uFghOgP6DjqIj_posthog=%7B%22distinct_id%22%3A%22019b52c3-8a21-7ddb-80d3-6705de899e5b%22%2C%22%24sesid%22%3A%5B1766623583834%2C%22019b52f8-41a6-7a9b-b61d-86aee0bb2210%22%2C1766623560100%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Ftokenterminal.com%2Fexplorer%2Fprojects%2Faave%2Ffinancial-statement%22%7D%7D',
#     }
#     data = rq.get(url, params, headers=headers).json()[0]['result']['data']

#     df = (
#         pd.DataFrame(data)
#         .pivot(index='timestamp', columns='metric_id', values='value')
#         .reset_index()
#     )
#     df['date'] = pd.to_datetime(df['timestamp'])
#     df = df[df['date'] < arrow.now('UTC').floor('month').datetime]
#     if len(df) < 12:
#         raise ValueError
#     df = df.sort_values('date')

#     def get_series(col_name):
#         return df.get(col_name, pd.Series(0, df.index))

#     supply = get_series('token_supply_circulating')
#     df['eps_ttm'] = get_series('earnings').rolling(12).sum() / supply
#     df['rps_ttm'] = get_series('fees').rolling(12).sum() / supply
#     df = df.iloc[11:]

#     d = (df['date'] + pd.DateOffset(months=1)).dt.date
#     zeros = pd.Series(0, df.index)
#     eps_ttm = get_series('eps_ttm')
#     rps_ttm = get_series('rps_ttm')
#     return [
#         Income(*_)
#         for _ in zip(
#             d, zeros, zeros, zeros, zeros, zeros, zeros, zeros, eps_ttm, rps_ttm
#         )
#     ]


def get_incomes(market: Literal['j', 't', 'u'], symbol: str, q: int):
    return get_incomes_from_fmp(market, symbol, q)


def create_sankey_frames(incomes: list[Income], q: int):
    incomes = incomes[-q:]
    max_r = max(e.r for e in incomes)
    frames = [
        go.Sankey(
            hoverinfo='skip',
            link=dict(
                color=[
                    TRANSPARENT,
                    TRANSPARENT,
                    LIGHT_RED,
                    LIGHT_GREEN if income.gp > 0 else LIGHT_RED,
                    LIGHT_RED,
                    LIGHT_GREEN if income.oi > 0 else LIGHT_RED,
                    LIGHT_RED,
                    LIGHT_RED,
                ],
                source=[0, 1, 2, 2, 4, 4, 5, 5],
                target=[1, 2, 3, 4, 5, 6, 7, 8],
                value=[
                    (abs(e) + 1) / 1e6
                    for e in (
                        max_r,
                        income.r,
                        income.cor,
                        income.gp,
                        income.oe,
                        income.oi,
                        income.rnd,
                        income.sgna,
                    )
                ],
            ),
            name=income.d.strftime('%y-%m-%d'),
            node=dict(
                color=[
                    TRANSPARENT,
                    TRANSPARENT,
                    DARK_GREEN,
                    DARK_RED,
                    DARK_GREEN if income.gp > 0 else DARK_RED,
                    DARK_RED,
                    DARK_GREEN if income.oi > 0 else DARK_RED,
                    DARK_RED,
                    DARK_RED,
                ],
                label=[
                    '',
                    '',
                    f'Revenue: {income.r // 1e6:,.0f}M',
                    f'Cost of Revenue: {income.cor // 1e6:,.0f}M',
                    f'Gross Profit: {income.gp // 1e6:,.0f}M',
                    f'Operating Expenses: {income.oe // 1e6:,.0f}M',
                    f'Operating Income: {income.oi // 1e6:,.0f}M',
                    f'R&D: {income.rnd // 1e6:,.0f}M',
                    f'SG&A: {income.sgna // 1e6:,.0f}M',
                ],
                line={'width': 0},
                x=[-0.67, -0.33, 0.01, 0.33, 0.33, 0.67, 0.67, 1.0, 1.0],
                y=[0.64, 0.64, 0.64, 1.0, 0.29, 0.57, 0.01, 0.34, 0.8],
            ),
        )
        for income in incomes + [incomes[-1]]
    ]
    frames[-1].name = 'Today'
    return frames


def get_prices(market: Literal['j', 't', 'u'], symbol: str, q: int, ema7: bool):
    prices = rq.get(
        f'http://localhost:8080/prices?market={market}&symbol={symbol}&n={91 * q}&ema7={"true" if ema7 else "false"}'
    ).json()
    today = pd.Timestamp.now(MARKET_TO_TIMEZONE[market]).date()
    date_index = pd.date_range(end=today, periods=len(prices), freq='D').date
    return pd.Series(prices, date_index)


def calc_bands(incomes: list[Income], prices: pd.Series, metric: str):
    s = (
        pd.Series({income.d: getattr(income, metric) for income in incomes})
        .reindex(pd.date_range(incomes[0].d, prices.index[-1]).date, method='ffill')
        .tail(len(prices))
    )
    if len(s) != len(prices) or (metric == 'rps' and pd.isna(s.iloc[0])):
        raise ValueError
    s[s <= 0] = None
    M = (prices / s).dropna()
    bands = pd.DataFrame(index=s.index)
    if M.empty:
        return bands
    for p in np.linspace(0, 1, 9):
        m = M.quantile(p)
        bands[m] = s * m
    # future = pd.date_range(bands.index[-1] + pd.Timedelta(days=1), periods=6)
    # return pd.concat([bands, pd.DataFrame([bands.iloc[-1]] * 6, future)])
    return bands


def create_price_frames_and_bands(
    market: Literal['j', 't', 'u'], symbol, incomes, q: int, ema7: bool
):
    prices = get_prices(market, symbol, q, ema7)
    dates = [e.d for e in incomes[-q:]] + [prices.index[-1]]
    frames = [
        go.Scatter(
            hoverlabel=dict(
                align='right', bgcolor='white', bordercolor='white', font=FONT
            ),
            hovertemplate='%{x|%y-%m-%d}<br>%{y:.2f}<extra></extra>',
            line=dict(color=BLUE, shape='spline', width=4),
            mode='lines',
            x=prices.index,
            y=prices.loc[: d + pd.Timedelta(days=7)],
        )
        for d in dates
    ]
    pe_df = calc_bands(incomes, prices, 'eps').fillna(0)
    pe_bands = [
        go.Scatter(
            fill='tonexty' if i else None,
            hoverinfo='skip',
            line=dict(color=BAND_COLORS[i], width=0),
            mode='lines',
            name=round(m),
            x=pe_df.index,
            y=band,
        )
        for i, (m, band) in enumerate(pe_df.items())
    ]
    ps_df = calc_bands(incomes, prices, 'rps').fillna(0)
    ps_bands = [
        go.Scatter(
            fill='tonexty' if i else None,
            hoverinfo='skip',
            line=dict(color=BAND_COLORS[i], width=0),
            mode='lines',
            name=round(m),
            x=ps_df.index,
            y=band,
        )
        for i, (m, band) in enumerate(ps_df.items())
    ]
    return frames, pe_bands, ps_bands


def get_displayed_fmp_url(market: Literal['j', 't', 'u'], symbol: str):
    if market in ('j', 't'):
        url = (
            f'https://financialmodelingprep.com/api/v3/income-statement/{add_suffix(market, symbol)}'
            f'?apikey={FMP_KEY}&limit=4&period=quarter'
        )
    else:
        url = (
            'https://financialmodelingprep.com/stable/income-statement'
            f'?apikey={FMP_KEY}&limit=4&period=quarter&symbol={symbol}'
        )
    return f'```\n{url}\n```'


@callback(
    Output('graph', 'figure'),
    Output('alert', 'displayed'),
    Output('fmp-url', 'children'),
    State('market', 'value'),
    State('input', 'value'),
    State('q', 'value'),
    State('ema7', 'value'),
    Input('button', 'n_clicks'),
)
def main(
    market: Literal['j', 't', 'u'],
    symbol: str,
    q: int,
    ema7_values: list[str],
    n_clicks: int,
):
    use_ema7 = 'on' in ema7_values
    fmp_url = get_displayed_fmp_url(market, symbol)
    if not (incomes := get_incomes(market, symbol, q)):
        return (
            go.Figure(go.Sankey(), go.Layout(paper_bgcolor=TRANSPARENT)),
            True,
            fmp_url,
        )
    s_frames = create_sankey_frames(incomes, q)
    p_frames, pe_bands, ps_bands = create_price_frames_and_bands(
        market, symbol, incomes, q, use_ema7
    )
    fig = make_subplots(
        3,
        1,
        specs=[[{'type': 'sankey'}], [{'type': 'xy'}], [{'type': 'xy'}]],
        vertical_spacing=0.1,
    )
    fig.add_trace(s_frames[-1], 1, 1)
    fig.add_trace(p_frames[-1], 2, 1)
    fig.add_trace(p_frames[-1], 3, 1)
    for band in pe_bands:
        fig.add_trace(band, 2, 1)
        fig.add_annotation(
            showarrow=False,
            text=band.name + 'x',
            x=band.x[-1],
            y=band.y[-1],
            row=2,
            col=1,
        )
    for band in ps_bands:
        fig.add_trace(band, 3, 1)
        fig.add_annotation(
            showarrow=False,
            text=band.name + 'x',
            x=band.x[-1],
            y=band.y[-1],
            row=3,
            col=1,
        )
    fig.frames = [
        go.Frame(data=[s_frame, p_frame, p_frame], name=s_frame.name)
        for s_frame, p_frame in zip(s_frames, p_frames)
    ]
    fig.update_layout(
        dict(
            font=FONT,
            paper_bgcolor=TRANSPARENT,
            plot_bgcolor=TRANSPARENT,
            showlegend=False,
            sliders=[
                dict(
                    borderwidth=0,
                    currentvalue={'visible': False},
                    len=0.92,
                    pad={'b': 40},
                    steps=[
                        dict(
                            args=[
                                [s_frame.name],
                                dict(mode='immediate', transition={'duration': 0}),
                            ],
                            label=s_frame.name,
                            method='animate',
                        )
                        for s_frame in s_frames
                    ],
                    tickcolor=FONT_COLOR,
                    x=0.12,
                )
            ],
            updatemenus=[
                dict(
                    bgcolor='white',
                    borderwidth=0,
                    buttons=[
                        dict(
                            args=[
                                None,
                                dict(
                                    frame={'duration': 2000},
                                    fromcurrent=True,
                                    transition={'duration': 0},
                                ),
                            ],
                            label='⏵',
                            method='animate',
                        ),
                        dict(
                            args=[[None], dict(mode='immediate')],
                            label='⏸',
                            method='animate',
                        ),
                    ],
                    direction='right',
                    font={'family': 'sans-serif'},
                    showactive=False,
                    type='buttons',
                    x=0.09,
                    y=-0.02,
                )
            ],
            xaxis1=dict(showgrid=False, visible=False),
            yaxis1=dict(showgrid=False, visible=False),
            xaxis2=dict(showgrid=False, visible=False),
            yaxis2=dict(showgrid=False, visible=False),
        )
    )
    return fig, False, fmp_url


if __name__ == '__main__':
    app.run(debug=True)
