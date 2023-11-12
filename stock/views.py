from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.views import View

from .ml import load_ds, model, algo, load_stock, model_arima, model_lstm, data_preprocess_lstm
from .models import UserActivity


def home(req: HttpRequest) -> HttpResponse:
    return render(req, "stock/home.html")

@method_decorator(login_required, name='dispatch')
class Home(View):
    template_name = "stock/home.html"
    forecast_days: int = 0
    user_input: str = ""
    model_name: str = ""
    predicted: float = float('-inf')

    def get(self, req: HttpRequest) -> HttpResponse:
        df = load_ds()
        symbol, name = list(df['Symbol']), list(df['Name'])
        context = {
            'stock_data': zip(symbol[1:], name[1:]),
            'default': {'symbol': symbol[0], 'name': name[0]},
        }
        return render(req, self.template_name, context)

    def post(self, req: HttpRequest) -> HttpResponse:
        print("HERE POSTED")
        forecast_days, user_input, model_name = int(req.POST['forecast_days']), req.POST['user_input'], req.POST['model_name']
        if forecast_days < 0:
            messages.error(req, "Forecast Cannot be Negative")
        UserActivity.objects.create(user=req.user, forecast_days=forecast_days, stock_symbol=user_input, model_name=model_name)
        
        if model_name == 'arima':
            self.predicted = model_arima(forecast_days, user_input)
        elif model_name == 'lstm':
            self.predicted = model_lstm(forecast_days, user_input)
        else:
            self.predicted = model(forecast_days, user_input, algo[model_name])
        
        df = load_ds()
        symbol, name = list(df['Symbol']), list(df['Name'])
        stock_info = load_stock(user_input)
        open, adj = list(stock_info['Open']), list(stock_info['Adj Close'])
        context = {
            'stock_data': zip(symbol[1:], name[1:]),
            'default': {'symbol': symbol[0], 'name': name[0]},
            'predicted': self.predicted,
            'stock_info': zip(open[:50], adj[:50]),
            'open_list': open,
            'adj_list': adj,
        }
        return render(req, self.template_name, context)